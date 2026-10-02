/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CE_COLL_H_
#define NCCL_CE_COLL_H_

#include "nccl.h"
#include "nccl_common.h"
#include "bitops.h"
#include "sym_kernels.h"
#include <algorithm>
#include <cassert>
#include <vector>

// Memory operations per rank for different synchronization protocols
#define NCCL_CE_SYNC_OPS_PER_RANK_MC 2
#define NCCL_CE_SYNC_OPS_PER_RANK_UC 3
#define RCCL_CE_NUM_COPY_STREAMS 8

// Selection marker for the hierarchical CE path. Defined once because the
// scale-out MPI tests assert on this exact text.
#define RCCL_CE_HIER_SELECTED_TAG "[Hierarchical CE]"

// Total payload capacity of one reusable CE AllReduce/ReduceScatter staging slot.
// Messages larger than this are pipelined.
#define NCCL_CE_AR_STAGING_BYTES (256ull * 1024 * 1024)

// Fallback 2-shot max cap for rcclCeAr2ShotMax() when no arch table is present.
// Independent of NCCL_CE_AR_STAGING_BYTES (which governs buffer allocation).
#ifndef NCCL_CE_AR_TMPBUF_DEFAULT_BYTES
#define NCCL_CE_AR_TMPBUF_DEFAULT_BYTES (256ULL * 1024 * 1024)
#endif

#ifndef NCCL_CE_REDUCE_MAX_BLOCKS
#define NCCL_CE_REDUCE_MAX_BLOCKS 92
#endif

#ifndef NCCL_CE_REDUCE_DEFAULT_BLOCKS
#define NCCL_CE_REDUCE_DEFAULT_BLOCKS 46
#endif

#ifndef NCCL_CE_NUM_SLOTS
#define NCCL_CE_NUM_SLOTS 2
#endif

#ifndef NCCL_CE_REDUCE_PER_CHUNK_SLOTS
#define NCCL_CE_REDUCE_PER_CHUNK_SLOTS 12
#endif

// Per-rank staging capacity in ceARTmpBuf (fixed default; use ceArStagingBytes for runtime value).
inline size_t ncclCeAllReduceMaxChunkBytes(int nRanks) {
  return (size_t)NCCL_CE_AR_STAGING_BYTES / (size_t)nRanks;
}

// Per-rank slot size in ceARTmpBuf. The host scatter addresses slots in bytes
// (rank * slotChunkBytes) while the reduce kernel addresses them in elements
// (rank * slotChunkElems), so a slot must hold a whole number of elements. 16 is
// a multiple of every supported element size and also keeps each rank boundary
// aligned for the kernel's 16B vector loads, so one round-down satisfies both.
inline size_t ncclCeAllReduceSlotChunkBytes(size_t maxChunkBytes) {
  return alignDown(maxChunkBytes, (size_t)16);
}

// Chunk size for the pipelined path, i.e. when a shard does not fit in one slot.
// A chunk must fit its slot, and needs the same 16B rounding as the slot itself:
// the host walks chunks in bytes (ch * chunkBytes) while the kernel walks them in
// elements (ch * baseChunkElems).
inline size_t ncclCeAllReduceChooseChunkBytes(size_t shardBytes, size_t slotChunkBytes) {
  const size_t MIN_CHUNK_BYTES = 4 * 1024 * 1024ULL;
  const size_t MAX_CHUNK_BYTES = 256 * 1024 * 1024ULL;
  size_t targetChunkBytes = shardBytes / 4;
  if (targetChunkBytes > MAX_CHUNK_BYTES) targetChunkBytes = MAX_CHUNK_BYTES;
  if (targetChunkBytes < MIN_CHUNK_BYTES) targetChunkBytes = MIN_CHUNK_BYTES;
  if (targetChunkBytes > slotChunkBytes) targetChunkBytes = slotChunkBytes;
  return alignDown(targetChunkBytes, (size_t)16);
}

// CE ReduceScatter staging offsets, split out of ncclCeReduceScatter() for host testability;
// ncclCeAllReduce() still open-codes these same three formulas.

// Offset is keyed by senderRank, not dstRank: every peer reduces the same slot layout.
inline size_t ncclCeReduceScatterDstSlotOffsetBytes(int slot, int senderRank, int nRanks, size_t slotChunkBytes) {
  return ((size_t)slot * (size_t)nRanks + (size_t)senderRank) * slotChunkBytes;
}

// Offset of chunk `chunk` within dstRank's shard in sendbuff (shardBytes = recvcount * eltSize).
inline size_t ncclCeReduceScatterSrcOffsetBytes(int dstRank, size_t shardBytes, int chunk, size_t chunkBytes) {
  return (size_t)dstRank * shardBytes + (size_t)chunk * chunkBytes;
}

// [slot][rank] doorbell index; must match between the local array-index view and the peer byte-offset view.
inline size_t ncclCeReduceScatterSignalIndex(int slot, int rank, int nRanks) {
  return (size_t)slot * (size_t)nRanks + (size_t)rank;
}

// ---------------------------------------------------------------------------
// Flat [sender][chunkOffset] ReduceScatter staging + ratio-guarded taper.
//
// The slotted layout above assumes every staged chunk is the same size
// (stride == slotChunkBytes, wrapping every NUM_SLOTS chunks). When a
// shard's full per-rank staging budget (numStagingSlots * ceArStagingBytes /
// nRanks) covers the shard with room to spare, each sender can instead own
// one contiguous per-rank region and address chunks by a running byte
// offset -- chunk COUNT and chunk SIZE become free parameters, with no slot
// reuse and no fixed stride (see ncclCeReduceScatter's three eligibility
// checks for the exact budget guards that gate this).
//
// That freedom unlocks a tapered schedule: large "body" chunks (each one's
// copy fully overlaps the *previous* chunk's reduce -- true for any
// uniform body size here, since the measured CE-copy byte rate kappa is
// several times the reducer's byte rate rho), then the shard's remainder
// split so the second-to-last piece's copy still hides the *last* piece's
// reduce. Only that final (small) reduce is ever exposed on the critical
// path, instead of a full body-sized one.
//
// ratioGuard = kappa/rho is the break-even point: a chunk's reduce is
// hidden behind the next chunk's copy iff the next chunk's bytes are at
// least 1/ratioGuard of the current chunk's bytes. Splitting a remainder M
// into [M*ratioGuard/(ratioGuard+1), M/(ratioGuard+1)] is the *smallest*
// trailing piece that still clears that bar for the piece ahead of it.
//
// rho depends on the reducer's actual launched thread width. The generated
// per-type launcher (src/device/ce_reduce/ce_reduce_launcher.cpp.in) is
// outside this task's editable file set and currently launches 256 threads
// for every type, so the constants below are calibrated for that width.
// kappa (the CE copy rate) does not depend on reducer width; only rho and
// ratioGuard would need to move if the launched width ever changes.
#define NCCL_CE_RS_TAPER_COPY_RATE_US_PER_MIB 25.9    // kappa: measured CE hipMemcpyBatchAsync byte rate.
#define NCCL_CE_RS_TAPER_COPY_OVERHEAD_US 30.0        // phi: fixed per-chunk copy/doorbell overhead.
#define NCCL_CE_RS_TAPER_REDUCE_RATE_US_PER_MIB 5.874 // rho: reducer byte rate AT THE LAUNCHED 256 THREADS.
#define NCCL_CE_RS_TAPER_REDUCE_OVERHEAD_US 3.3        // reducer fixed per-chunk overhead.
#define NCCL_CE_RS_TAPER_DOORBELL_US 11.0              // hipStreamBatchMemOp doorbell, per chunk.
#define NCCL_CE_RS_TAPER_RATIO_GUARD \
  (NCCL_CE_RS_TAPER_COPY_RATE_US_PER_MIB / NCCL_CE_RS_TAPER_REDUCE_RATE_US_PER_MIB)
#define NCCL_CE_RS_TAPER_MIN_TAIL_BYTES (1ull * 1024 * 1024) // do not split off a taper sliver below ~1 MiB.

// Splits `remainder` into at most two 16B-aligned pieces summing to exactly
// `remainder`: [remainder-tail, tail], tail being the smallest trailing
// piece whose predecessor's reduce is still hidden behind tail's own copy
// time (see ratioGuard above). Falls back to one piece when the computed
// tail would be smaller than minTailBytes (not worth a second doorbell) or
// when remainder itself is already that small. Pure/host-testable.
inline void ncclCeReduceScatterTaperRemainder(size_t remainder, double ratioGuard, size_t minTailBytes,
                                               std::vector<size_t>& outPieces) {
  if (remainder == 0) return;
  const size_t tail = alignDown((size_t)((double)remainder / (ratioGuard + 1.0)), (size_t)16);
  if (tail > 0 && tail >= minTailBytes && tail < remainder) {
    outPieces.push_back(remainder - tail);
    outPieces.push_back(tail);
  } else {
    outPieces.push_back(remainder);
  }
}

// Builds the full per-shard schedule: floor(shardBytes/bodyBytes) chunks of
// bodyBytes, then the tapered remainder. If shardBytes is an exact multiple
// of bodyBytes, one body chunk is folded back into the remainder so the
// taper still has a predecessor to protect -- otherwise the very last body
// chunk's reduce would be fully exposed with nothing after it to hide it
// behind. By construction (bodyBytes and shardBytes both 16B-aligned, tail
// always aligned down to 16) this is an exact, 16B-aligned cover of
// shardBytes; asserted here as a cheap, self-documenting invariant and
// re-verified with a plain runtime check at the call site in
// ncclCeReduceScatter (which does not rely on NDEBUG for its safety).
inline void ncclCeReduceScatterBuildTaperSchedule(size_t shardBytes, size_t bodyBytes, double ratioGuard,
                                                   size_t minTailBytes, std::vector<size_t>& schedule) {
  schedule.clear();
  if (shardBytes == 0) return;
  if (bodyBytes == 0 || bodyBytes >= shardBytes) {
    ncclCeReduceScatterTaperRemainder(shardBytes, ratioGuard, minTailBytes, schedule);
  } else {
    size_t nBody = shardBytes / bodyBytes;
    size_t remainder = shardBytes - nBody * bodyBytes;
    if (remainder == 0) {
      nBody--;
      remainder = bodyBytes;
    }
    for (size_t i = 0; i < nBody; i++) schedule.push_back(bodyBytes);
    ncclCeReduceScatterTaperRemainder(remainder, ratioGuard, minTailBytes, schedule);
  }
#ifndef NDEBUG
  size_t cover = 0;
  for (size_t s : schedule) {
    assert(s % 16 == 0 && "taper schedule piece is not 16B-aligned");
    cover += s;
  }
  assert(cover == shardBytes && "taper schedule does not exactly cover shardBytes");
#endif
}

// Predicts total pipeline latency (us) for `schedule` via the critical-path
// recurrence C[k]=C[k-1]+copy(s_k), R[k]=max(C[k],R[k-1])+reduce(s_k),
// total=R[last]+doorbellUs*chunkCount, with copy(s)=kappa*MiB(s)+copyOverhead
// and reduce(s)=rho*MiB(s)+reduceOverhead. Pure/host-testable. Used only to
// RANK candidate body sizes in ncclCeReduceScatterChooseSchedule below --
// never consulted for correctness, so an imprecise constant only costs a
// fraction of a percent of ranking quality, never a wrong result.
inline double ncclCeReduceScatterPredictUs(const std::vector<size_t>& schedule, double kappaUsPerMiB,
                                            double copyOverheadUs, double rhoUsPerMiB, double reduceOverheadUs,
                                            double doorbellUs) {
  const double MiB = 1024.0 * 1024.0;
  double cPrev = 0.0, rPrev = 0.0;
  for (size_t s : schedule) {
    const double mib = (double)s / MiB;
    const double c = cPrev + kappaUsPerMiB * mib + copyOverheadUs;
    const double r = std::max(c, rPrev) + rhoUsPerMiB * mib + reduceOverheadUs;
    cPrev = c;
    rPrev = r;
  }
  return rPrev + doorbellUs * (double)schedule.size();
}

// Picks the body size, from a small fixed candidate list, that minimizes
// ncclCeReduceScatterPredictUs for this exact shardBytes, and returns its
// schedule. This keeps the body size a *measured* choice -- goal.md's own
// finding is that it is shape-dependent ("search B instead of hard-coding
// one constant") -- rather than freezing a single value that was
// calibrated for a different reducer thread width than the one actually in
// effect. Bounded to a handful of candidates, so this stays O(1) and
// allocation-light per call.
inline void ncclCeReduceScatterChooseSchedule(size_t shardBytes, double ratioGuard, size_t minTailBytes,
                                               double kappaUsPerMiB, double copyOverheadUs, double rhoUsPerMiB,
                                               double reduceOverheadUs, double doorbellUs,
                                               std::vector<size_t>& schedule) {
  static const size_t kBodyCandidatesBytes[] = {
    64ull * 1024 * 1024,  96ull * 1024 * 1024,  128ull * 1024 * 1024,
    160ull * 1024 * 1024, 192ull * 1024 * 1024, 224ull * 1024 * 1024,
  };
  schedule.clear();
  double bestUs = -1.0;
  std::vector<size_t> candidate;
  for (size_t bodyBytes : kBodyCandidatesBytes) {
    ncclCeReduceScatterBuildTaperSchedule(shardBytes, bodyBytes, ratioGuard, minTailBytes, candidate);
    const double us = ncclCeReduceScatterPredictUs(candidate, kappaUsPerMiB, copyOverheadUs, rhoUsPerMiB,
                                                   reduceOverheadUs, doorbellUs);
    if (bestUs < 0.0 || us < bestUs) {
      bestUs = us;
      schedule = candidate;
    }
  }
}

enum ncclCeMethodId {
  ncclCeMethodId_AllGather_UC,
  ncclCeMethodId_AllGather_MC,
  ncclCeMethodId_Count
};

struct ncclCeColl {
  bool initialized;
  uint8_t* baseUCSymReadyPtr;
  uint8_t* baseUCSymComplPtr;
  size_t baseUCSymReadyOffset;
  size_t baseUCSymComplOffset;
  uint32_t ceSeqNum;
  // Device buffer sourcing the UC barrier flag value. Slot [0]: running seq
  // (stored per non-capture barrier); slot [1]: constant GRAPH_SYNC_VALUE used
  // during graph capture. Peer writes memcpy from here so they can be issued
  // as a separate stream op ahead of the wait/reset batch (see ncclPrepUCSync).
  uint32_t* ceSeqNumDev;
  bool useCompletePtr;
  uint32_t intraBatchSyncFreq;
  uint64_t intraBatchSyncMsgThreshold;
  int64_t agMulticastThreshold;  // user override (>=0); -1 -> use cost model
  struct ncclDevrWindow* ceSyncWin;
  int nCopyStreams;
  cudaStream_t copyStreams[RCCL_CE_NUM_COPY_STREAMS];
  cudaEvent_t copyEvents[RCCL_CE_NUM_COPY_STREAMS];
#ifdef ENABLE_FAULT_INJECTION
  uint32_t ceFaults;  // bitmask of CE_FAULT_* bits; see ce_fault_inject.h
#endif

  // CE AllReduce/ReduceScatter staging buffer (symmetric). The default path is
  // double-buffered; opt-in per-chunk ReduceScatter reserves additional slots
  // so more staged chunks can use fresh slots before reuse.
  // The reduced result is written straight into the user recvbuff (no scratch).
  uint8_t* ceARTmpBuf;
  struct ncclDevrWindow* ceARTmpWin;
  size_t ceArMaxBytes;     // 2-shot staging cap, resolved at init: env RCCL_CE_AR_MAX_MSG_BYTES > arch ceArMax
  size_t ceArStagingBytes; // resolved at init: env var RCCL_CE_AR_STAGING_BYTES > NCCL_CE_AR_STAGING_BYTES
  size_t numStagingSlots;  // NCCL_CE_NUM_SLOTS, or the per-chunk RS slot count when that mode is enabled
  uint32_t* signalBuffer;
  struct ncclDevrWindow* signalWin;
  // Global counter barrier for regular launch: [0]=arrival, [1]=completed generation.
  uint32_t* d_barrierSync;
  cudaStream_t scatterStream;
  cudaEvent_t synceEvent;  // join scatterStream back onto the caller's stream
  // Latched while this comm has live graph-captured plans. CE 2-shot AllReduce
  // can deadlock on eager calls that share a graph-mode comm, so we disable CE
  // AR during that period and re-enable it after captured plans are reclaimed.
  // Written only from rcclCeAllReduceGraphLatchTick(); no internal lock, same
  // single-writer-per-comm contract as localPersistentRefs (comm.h).
  bool graphModeSeen;
};

struct ncclCeInitTask {
  struct ncclCeInitTask* next;
  struct ncclComm* comm;
};

struct alignas(16) ncclCeCollArgs {
  ncclFunc_t func;
  int rootRank;
  ncclDataType_t datatype;
  size_t nElts;
  size_t eltSize;
  uint8_t* sendBuff;
  uint8_t* recvBuff;
  struct ncclDevrWindow* sendWin;
  struct ncclDevrWindow* recvWin;

  // AlltoAllv: [sendSizes, sendDispls, recvSizes, recvDispls] x nRanks (bytes).
  size_t* sizes;

  void* collApiEventHandle;  // Parent API event handle for profiler hierarchy
  void* ceCollProfHandle;    // CE collective profiler event handle
  uint64_t userTag;          // Per-call profiler annotation (0 == untagged)
  bool useDda;
  void** ddaPeerBases;      // host-side table of every rank's DDA scratch base pointer
  void*
    ddaUserRecvBuff; // user recvbuff (using DDA staging) or NULL otherwise (if recvbuffer is using symmetric windows)
  size_t ddaCopyBackBytes; // bytes to copy scratch -> user recvbuff
  ncclRedOp_t redOp; // Used for AllReduce and ReduceScatter
};

struct ncclCeBatchOpsParams {
  void** dsts;
  void** srcs;
  size_t* sizes;
  size_t numOps;
  bool intraBatchSync;
#ifdef CE_BATCH_ASYNC_SUPPORTED
  hipMemcpyAttributes* attrs;
  size_t* attrIdxs;
  size_t numAttrs;
#endif
};

bool ncclCeAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                     ncclSymRegType_t winRegType, struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin);

bool ncclCeScratchAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                            ncclSymRegType_t winRegType);

bool ncclCeImplemented(ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty);

bool ncclHierCeAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                         ncclSymRegType_t winRegType, struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin);

// True when an admitted CE task must run on the hierarchical path. The launch
// site and the selection marker share this so they cannot disagree about which
// algorithm a task got.
bool ncclHierCeDispatch(struct ncclComm* comm);

ncclResult_t ncclCeInit(struct ncclComm* comm);

ncclResult_t ncclCeFinalize(struct ncclComm* comm);

// Intra-LSA-rank barrier.
ncclResult_t ncclMemOpSync(struct ncclComm* comm, cudaStream_t stream, struct ncclCeCollArgs* profilerArgs = nullptr);

// Allocate / free internal arrays for a batch-ops parameter struct.
ncclResult_t ncclCeInitBatchOpsParams(struct ncclCeBatchOpsParams* params, int capacity);
void ncclCeFreeBatchOpsParams(struct ncclCeBatchOpsParams* params);

// Launch a batch of cudaMemcpyAsync ops
ncclResult_t ncclCeLaunchBatchOps(struct ncclComm* comm, struct ncclCeBatchOpsParams* params, cudaStream_t stream,
                                  struct ncclCeCollArgs* profilerArgs = nullptr);

ncclResult_t ncclLaunchCeColl(struct ncclComm* comm, struct ncclKernelPlan* plan);

ncclResult_t scheduleCeCollTaskToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan);

ncclResult_t ncclCeAllGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

int ncclCeAllGatherUseMulticast(struct ncclComm* comm, size_t perRankBytes, int captured, int inPlace);

ncclResult_t ncclCeScatter(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclCeGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclCeAlltoAll(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclCeAlltoAllv(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclAlltoAllvValidatePeerSendSize(size_t sendBytes, size_t peerRecvBytes, int srcRank, int dstRank);

bool ncclCeAlltoAllvEligible(struct ncclComm* comm, ncclDataType_t datatype, ncclSymRegType_t winRegType,
                             bool hasSysmemSegment, bool capturing);

// Same gates as AlltoAllv, then ncclCeAvailable (single-node CE; not hier).
bool ncclCeAlltoAllEligible(struct ncclComm* comm, ncclDataType_t datatype, ncclSymRegType_t winRegType,
                            bool hasSysmemSegment, bool capturing);

ncclResult_t ncclHierCeAllGather(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);

ncclResult_t ncclHierCeAlltoAll(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);

// CE AllReduce: scatter → local-reduce → allgather (→ optional copy-to-user-recvbuff).
// Requires comm->ceColl.ceARTmpBuf != NULL (i.e. ncclCeInit has run).
ncclResult_t ncclCeAllReduce(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                             ncclDataType_t datatype, ncclRedOp_t op, cudaStream_t stream,
                             struct ncclDevrWindow* recvWin = nullptr,
                             struct ncclCeCollArgs* profilerArgs = nullptr);

// CE ReduceScatter: scatter → local-reduce into recvbuff (recvcount elements per rank).
// Uses the same staging buffer as CE AllReduce (ncclCeEnsureAllReduceStaging).
ncclResult_t ncclCeReduceScatter(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount,
                                 ncclDataType_t datatype, ncclRedOp_t op, cudaStream_t stream,
                                 struct ncclDevrWindow* recvWin = nullptr,
                                 struct ncclCeCollArgs* profilerArgs = nullptr);

// Reduce-kernel block count for a per-rank chunk of `chunkElems` elements
// (chunkElems = count / nRanks). Mirrors the geometry ncclCeLaunchLocalReduce
// launches; for host-side impl-selection reporting. Returns 0 if chunkElems==0.
int ncclCeLocalReduceBlocks(ncclDataType_t datatype, size_t chunkElems);
#endif /* NCCL_CE_COLL_H_ */
