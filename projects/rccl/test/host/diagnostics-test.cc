/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ras/diagnostics.cc.

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#define rasClientsHead DiagnosticsTestRasClientsHead
#define rasCollFree DiagnosticsTestRasCollFree
#define rasCollReqInit DiagnosticsTestRasCollReqInit
#define rasNetSendCollReq DiagnosticsTestRasNetSendCollReq
#define rasDiagnosticsGpuModelCollectLocal DiagnosticsTestGpuModelCollectLocal
#define rasDiagnosticsGpuModelSummarize DiagnosticsTestGpuModelSummarize
#define rasDiagnosticsCudaDriverVersionCollectLocal DiagnosticsTestCudaDriverVersionCollectLocal
#define rasDiagnosticsCudaDriverVersionSummarize DiagnosticsTestCudaDriverVersionSummarize
#define rasDiagnosticsEccCollectLocal DiagnosticsTestEccCollectLocal
#define rasDiagnosticsEccSummarize DiagnosticsTestEccSummarize
#define rasDiagnosticsNvLinkCollectLocal DiagnosticsTestNvLinkCollectLocal
#define rasDiagnosticsNvLinkSummarize DiagnosticsTestNvLinkSummarize
#define rasDiagnosticsNcclEnvCollectLocal DiagnosticsTestNcclEnvCollectLocal
#define rasDiagnosticsNcclEnvSummarize DiagnosticsTestNcclEnvSummarize

#include "comm.h"
#include "ras/diagnostics.h"
#include "ras/diagnostics_checks.h"
#include "ras/ras_internal.h"

#define clockNano DiagnosticsTestClockNano

uint64_t DiagnosticsTestClockNano();

#include DIAGNOSTICS_CC_PATH

#undef clockNano

namespace {
int64_t g_clockNano = 0;
}  // namespace

uint64_t DiagnosticsTestClockNano() { return static_cast<uint64_t>(g_clockNano); }

// Test-local collaborator fakes.

struct rasClient* rasClientsHead = nullptr;

namespace {
int g_collFreeCalls = 0;
struct rasCollective* g_lastCollFree = nullptr;

uint64_t g_collReqLastRootId = 0;

int g_netSendCollReqCalls = 0;
ncclResult_t g_netSendCollReqResult = ncclSuccess;
bool g_netSendCollReqAllDone = true;
struct rasCollective* g_netSendCollReqCollToAssign = nullptr;
struct rasCollRequest g_lastSentReq{};

struct CheckHook {
  ncclResult_t collectLocalResult = ncclSuccess;
  struct rasDiagnosticsLocalData collectLocalData{};
  int collectLocalCalls = 0;

  ncclResult_t summarizeResult = ncclSuccess;
  int summarizeCalls = 0;
  // Copied, not a raw pointer: rasDiagnosticsSummarizePeerPayloads frees its
  // combined[id].records buffer immediately after this call returns.
  std::vector<char> lastSummarizeData;
  int lastSummarizeNData = -1;
  const struct rasDiagnosticsContext* lastSummarizeCtx = nullptr;
};
CheckHook g_checkHooks[RAS_DIAG_CHECK_COUNT];

void ResetCheckHooks() {
  for (auto& h : g_checkHooks) h = CheckHook{};
}
}  // namespace

void rasCollFree(struct rasCollective* coll) {
  ++g_collFreeCalls;
  g_lastCollFree = coll;
  if (coll == nullptr) return;
  free(coll->fwdConns);
  free(coll->peers);
  free(coll->data);
  free(coll);
}

void rasCollReqInit(struct rasCollRequest* req) { req->rootId = ++g_collReqLastRootId; }

ncclResult_t rasNetSendCollReq(const struct rasCollRequest* req, bool* pAllDone, struct rasCollective** pColl,
                               struct rasConnection*) {
  ++g_netSendCollReqCalls;
  g_lastSentReq = *req;
  if (g_netSendCollReqResult != ncclSuccess) return g_netSendCollReqResult;
  if (pAllDone) *pAllDone = g_netSendCollReqAllDone;
  if (pColl) *pColl = g_netSendCollReqCollToAssign;
  return ncclSuccess;
}

#define DEFINE_CHECK_FAKE(idx, collectFn, summarizeFn)                                                            \
  ncclResult_t collectFn(const struct rasDiagnosticsContext*, struct rasDiagnosticsLocalData* data) {             \
    ++g_checkHooks[idx].collectLocalCalls;                                                                        \
    *data = g_checkHooks[idx].collectLocalData;                                                                   \
    return g_checkHooks[idx].collectLocalResult;                                                                  \
  }                                                                                                                \
  ncclResult_t summarizeFn(const struct rasDiagnosticsContext* ctx, const struct rasDiagnosticsReporter*,          \
                           const char* data, int nData) {                                                          \
    ++g_checkHooks[idx].summarizeCalls;                                                                            \
    g_checkHooks[idx].lastSummarizeCtx = ctx;                                                                      \
    if (data != nullptr && nData > 0) g_checkHooks[idx].lastSummarizeData.assign(data, data + nData);              \
    else g_checkHooks[idx].lastSummarizeData.clear();                                                              \
    g_checkHooks[idx].lastSummarizeNData = nData;                                                                  \
    return g_checkHooks[idx].summarizeResult;                                                                      \
  }

DEFINE_CHECK_FAKE(0, rasDiagnosticsGpuModelCollectLocal, rasDiagnosticsGpuModelSummarize)
DEFINE_CHECK_FAKE(1, rasDiagnosticsCudaDriverVersionCollectLocal, rasDiagnosticsCudaDriverVersionSummarize)
DEFINE_CHECK_FAKE(2, rasDiagnosticsEccCollectLocal, rasDiagnosticsEccSummarize)
DEFINE_CHECK_FAKE(3, rasDiagnosticsNvLinkCollectLocal, rasDiagnosticsNvLinkSummarize)
DEFINE_CHECK_FAKE(4, rasDiagnosticsNcclEnvCollectLocal, rasDiagnosticsNcclEnvSummarize)

#undef DEFINE_CHECK_FAKE

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

namespace {

struct rasClient* MakeClient() {
  auto* client = static_cast<struct rasClient*>(calloc(1, sizeof(struct rasClient)));
  client->status = RAS_CLIENT_CONNECTED;
  return client;
}

void LinkClient(struct rasClient* client) {
  client->next = rasClientsHead;
  if (rasClientsHead) rasClientsHead->prev = client;
  rasClientsHead = client;
}

void FreeClientList() {
  struct rasClient* client = rasClientsHead;
  while (client) {
    struct rasClient* next = client->next;
    free(client->diagnostics);
    if (client->coll) rasCollFree(client->coll);
    free(client);
    client = next;
  }
  rasClientsHead = nullptr;
}

int g_emitCalls;
std::vector<std::string> g_emittedLines;
ncclResult_t g_finishResult;
int g_finishCalls;
ncclResult_t g_lastFinishResult;
void* g_lastFinishTarget;

ncclResult_t RecordingEmit(void* /*target*/, const char* line) {
  ++g_emitCalls;
  g_emittedLines.emplace_back(line);
  return ncclSuccess;
}

ncclResult_t RecordingFinish(void* target, ncclResult_t result) {
  ++g_finishCalls;
  g_lastFinishTarget = target;
  g_lastFinishResult = result;
  return g_finishResult;
}

void ResetRecordingReporter() {
  g_emitCalls = 0;
  g_emittedLines.clear();
  g_finishResult = ncclSuccess;
  g_finishCalls = 0;
  g_lastFinishResult = ncclSuccess;
  g_lastFinishTarget = nullptr;
}

rasDiagnosticsReporter MakeRecordingReporter() {
  rasDiagnosticsReporter reporter{};
  reporter.emit = RecordingEmit;
  reporter.finish = RecordingFinish;
  reporter.target = nullptr;
  return reporter;
}

struct FakeComm {
  std::unique_ptr<ncclComm> comm{new ncclComm{}};
  std::vector<ncclPeerInfo> peerInfos;

  FakeComm(uint64_t commHash, int nRanks) {
    comm->commHash = commHash;
    comm->nRanks = nRanks;
    comm->peerInfoValid = true;
    peerInfos.assign(nRanks > 0 ? nRanks : 1, ncclPeerInfo{});
    peerInfos[0].hostHash = commHash + 1;
    peerInfos[0].pidHash = commHash + 2;
    comm->peerInfo = peerInfos.data();
  }
  ncclComm* get() { return comm.get(); }
};

// Builds one peer's self-delimiting payload: [rasDiagnosticsPeerPayloadHeader]
// followed by one [rasDiagnosticsCheckPayloadHeader][records] block per contribution.
// Mirrors rasDiagnosticsAppendCheckPayload, which omits the header entirely for a
// check that produced zero records.
struct CheckContribution {
  rasDiagnosticsCheckId checkId;
  int recordStride;
  std::vector<char> records;
};

std::vector<char> BuildPeerPayload(const std::vector<CheckContribution>& checks) {
  std::vector<char> buf(sizeof(struct rasDiagnosticsPeerPayloadHeader), 0);
  int nChecks = 0;
  for (auto& c : checks) {
    struct rasDiagnosticsCheckPayloadHeader h{};
    h.checkId = c.checkId;
    h.recordStride = c.recordStride;
    h.nRecords = c.recordStride > 0 ? static_cast<int>(c.records.size() / c.recordStride) : 0;
    h.payloadBytes = static_cast<int>(c.records.size());
    size_t off = buf.size();
    buf.resize(off + sizeof(h));
    memcpy(buf.data() + off, &h, sizeof(h));
    buf.insert(buf.end(), c.records.begin(), c.records.end());
    nChecks++;
  }
  auto* hdr = reinterpret_cast<struct rasDiagnosticsPeerPayloadHeader*>(buf.data());
  hdr->nChecks = nChecks;
  hdr->payloadBytes = static_cast<int>(buf.size());
  return buf;
}

std::vector<char> BuildGatheredData(const std::vector<std::vector<CheckContribution>>& peers) {
  std::vector<char> all;
  for (auto& p : peers) {
    auto buf = BuildPeerPayload(p);
    all.insert(all.end(), buf.begin(), buf.end());
  }
  return all;
}

struct rasDiagnosticsCheckPayloadHeader* FirstCheckHeader(std::vector<char>& payload) {
  return reinterpret_cast<struct rasDiagnosticsCheckPayloadHeader*>(
    payload.data() + sizeof(struct rasDiagnosticsPeerPayloadHeader));
}

void ResetWholeFileSeams() {
  g_collFreeCalls = 0;
  g_lastCollFree = nullptr;
  g_collReqLastRootId = 0;
  g_netSendCollReqCalls = 0;
  g_netSendCollReqResult = ncclSuccess;
  g_netSendCollReqAllDone = true;
  g_netSendCollReqCollToAssign = nullptr;
  memset(&g_lastSentReq, 0, sizeof(g_lastSentReq));
  ResetCheckHooks();
  ResetRecordingReporter();
  FreeClientList();
  g_clockNano = 0;
}

class RasDiagnosticsMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { ResetWholeFileSeams(); }
  void TearDown() override { ResetWholeFileSeams(); }
};

}  // namespace

// ===========================================================================
// rasDiagnosticsFormatLine
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, FormatLine_NullArgsReturnInternalError) {
  char buf[64];
  EXPECT_EQ(ncclInternalError, rasDiagnosticsFormatLine(nullptr, sizeof(buf), "line"));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsFormatLine(buf, 0, "line"));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsFormatLine(buf, sizeof(buf), nullptr));
}

TEST_F(RasDiagnosticsMicrotest, FormatLine_TruncationReturnsInternalError) {
  char buf[8];
  EXPECT_EQ(ncclInternalError, rasDiagnosticsFormatLine(buf, sizeof(buf), "a much too long diagnostics line"));
}

TEST_F(RasDiagnosticsMicrotest, FormatLine_SuccessIncludesHostAndPid) {
  char buf[256];
  ASSERT_EQ(ncclSuccess, rasDiagnosticsFormatLine(buf, sizeof(buf), "hello"));
  EXPECT_NE(nullptr, strstr(buf, "NCCL DIAG hello"));
}

TEST_F(RasDiagnosticsMicrotest, FormatLine_ExactFitSucceedsAtTheBoundary) {
  char big[256];
  ASSERT_EQ(ncclSuccess, rasDiagnosticsFormatLine(big, sizeof(big), "hello"));
  size_t exactLen = strlen(big) + 1;  // Room for the line plus the terminating NUL.
  auto exact = std::make_unique<char[]>(exactLen);
  EXPECT_EQ(ncclSuccess, rasDiagnosticsFormatLine(exact.get(), exactLen, "hello"));
  EXPECT_STREQ(big, exact.get());
  // One byte short of that exact fit must fail (snprintf's return equals outSize).
  auto short_ = std::make_unique<char[]>(exactLen - 1);
  EXPECT_EQ(ncclInternalError, rasDiagnosticsFormatLine(short_.get(), exactLen - 1, "hello"));
}

// ===========================================================================
// rasDiagnosticsContextInit
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, ContextInit_NullContextReturnsInternalError) {
  EXPECT_EQ(ncclInternalError, rasDiagnosticsContextInit(nullptr, nullptr));
}

TEST_F(RasDiagnosticsMicrotest, ContextInit_NullCommZeroesUnscopedContext) {
  struct rasDiagnosticsContext ctx;
  memset(&ctx, 0xAB, sizeof(ctx));
  ASSERT_EQ(ncclSuccess, rasDiagnosticsContextInit(&ctx, nullptr));
  EXPECT_FALSE(ctx.hasCommFilter);
  EXPECT_EQ(0, ctx.commNRanks);
}

TEST_F(RasDiagnosticsMicrotest, ContextInit_InvalidPeerInfoReturnsInternalError) {
  FakeComm fc(0x1000, 4);
  fc.get()->peerInfoValid = false;
  struct rasDiagnosticsContext ctx{};
  EXPECT_EQ(ncclInternalError, rasDiagnosticsContextInit(&ctx, fc.get()));
}

TEST_F(RasDiagnosticsMicrotest, ContextInit_ValidCommCopiesFilterFields) {
  FakeComm fc(0x2000, 4);
  struct rasDiagnosticsContext ctx{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsContextInit(&ctx, fc.get()));
  EXPECT_TRUE(ctx.hasCommFilter);
  EXPECT_EQ(0x2000u, ctx.commFilter.commHash);
  EXPECT_EQ(0x2001u, ctx.commFilter.hostHash);
  EXPECT_EQ(0x2002u, ctx.commFilter.pidHash);
  EXPECT_EQ(4, ctx.commNRanks);
}

// ===========================================================================
// Static payload helpers
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, GetCheck_NullOutputReturnsInternalError) {
  EXPECT_EQ(ncclInternalError, rasDiagnosticsGetCheck(RAS_DIAG_CHECK_GPU_MODEL, nullptr));
}

TEST_F(RasDiagnosticsMicrotest, AppendData_RejectsInvalidSizesAndPointers) {
  char* data = nullptr;
  int nData = 0;
  const char byte = 1;

  EXPECT_EQ(ncclInternalError, rasDiagnosticsAppendData(nullptr, &nData, &byte, 1));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsAppendData(&data, nullptr, &byte, 1));
  nData = -1;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsAppendData(&data, &nData, &byte, 1));
  nData = 0;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsAppendData(&data, &nData, &byte, -1));
  nData = INT_MAX;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsAppendData(&data, &nData, &byte, 1));
  nData = 0;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsAppendData(&data, &nData, nullptr, 1));
}

TEST_F(RasDiagnosticsMicrotest, ValidateLocalData_RejectsInvalidMetadata) {
  struct rasDiagnosticsLocalData data{};
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, nullptr));

  data.recordsBytes = -1;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));
  data = {};
  data.recordStride = -1;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));
  data = {};
  data.nRecords = -1;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));

  char record = 0;
  data = {};
  data.records = &record;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));
  data = {};
  data.recordsBytes = 1;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));

  data = {};
  data.nRecords = 1;
  data.recordStride = 1;
  data.recordsBytes = 1;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));
  data.records = &record;
  data.recordStride = 0;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));
  data.recordStride = 2;
  data.nRecords = INT_MAX;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));
  data.nRecords = 1;
  data.recordsBytes = 1;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsValidateLocalData(RAS_DIAG_CHECK_GPU_MODEL, &data));
}

TEST_F(RasDiagnosticsMicrotest, CollectLocalPeerPayload_RejectsInvalidOutputsAndContext) {
  struct rasDiagnosticsContext ctx{};
  char* data = reinterpret_cast<char*>(1);
  int nData = 7;

  EXPECT_EQ(ncclInternalError, rasDiagnosticsCollectLocalPeerPayload(&ctx, nullptr, &nData));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsCollectLocalPeerPayload(&ctx, &data, nullptr));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsCollectLocalPeerPayload(nullptr, &data, &nData));
  EXPECT_EQ(nullptr, data);
  EXPECT_EQ(0, nData);
}

TEST_F(RasDiagnosticsMicrotest, AccountCheckRecords_RejectsStrideAndSizeOverflow) {
  struct rasDiagnosticsLocalData combined{};
  struct rasDiagnosticsCheckPayloadHeader header{};
  header.checkId = RAS_DIAG_CHECK_GPU_MODEL;
  header.recordStride = 4;
  header.nRecords = 1;
  header.payloadBytes = 4;

  combined.nRecords = 1;
  combined.recordStride = 8;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsAccountCheckRecords(&combined, &header));

  combined = {};
  combined.recordStride = 4;
  combined.nRecords = INT_MAX;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsAccountCheckRecords(&combined, &header));

  combined = {};
  combined.recordStride = 4;
  combined.recordsBytes = INT_MAX;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsAccountCheckRecords(&combined, &header));
}

// ===========================================================================
// rasDiagnosticsClientInit / rasDiagnosticsClientCleanup
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, ClientInit_NullClientOrCtxReturnsInternalError) {
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeClient();
  EXPECT_EQ(ncclInternalError, rasDiagnosticsClientInit(nullptr, &ctx, nullptr));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsClientInit(client, nullptr, nullptr));
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, ClientInit_ReporterWithNullEmitReturnsInternalError) {
  struct rasDiagnosticsContext ctx{};
  struct rasDiagnosticsReporter reporter{};
  auto* client = MakeClient();
  EXPECT_EQ(ncclInternalError, rasDiagnosticsClientInit(client, &ctx, &reporter));
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, ClientInit_ExistingDiagnosticsStateReturnsInternalError) {
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeClient();
  client->diagnostics = reinterpret_cast<struct rasDiagnosticsClientState*>(0x1);
  EXPECT_EQ(ncclInternalError, rasDiagnosticsClientInit(client, &ctx, nullptr));
  client->diagnostics = nullptr;  // Avoid a bogus free in cleanup.
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, ClientInit_NullReporterUsesDefault) {
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeClient();
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, nullptr));
  ASSERT_NE(nullptr, client->diagnostics);
  EXPECT_NE(nullptr, client->diagnostics->reporter.emit);
  rasDiagnosticsClientCleanup(client);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, ClientInit_CustomReporterAndContextAreStored) {
  struct rasDiagnosticsContext ctx{};
  ctx.hasCommFilter = true;
  ctx.commFilter.commHash = 0x42;
  auto reporter = MakeRecordingReporter();
  auto* client = MakeClient();
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, &reporter));
  EXPECT_EQ(RecordingEmit, client->diagnostics->reporter.emit);
  EXPECT_EQ(0x42u, client->diagnostics->ctx.commFilter.commHash);
  rasDiagnosticsClientCleanup(client);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, ClientCleanup_NullClientIsNoOp) { rasDiagnosticsClientCleanup(nullptr); }

TEST_F(RasDiagnosticsMicrotest, ClientCleanup_FreesAndNullsDiagnostics) {
  struct rasDiagnosticsContext ctx{};
  auto* client = MakeClient();
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, nullptr));
  rasDiagnosticsClientCleanup(client);
  EXPECT_EQ(nullptr, client->diagnostics);
  free(client);
}

// ===========================================================================
// rasDiagnosticsInProgress / rasDiagnosticsCancelTarget
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, InProgress_EmptyClientListIsFalse) { EXPECT_FALSE(rasDiagnosticsInProgress()); }

TEST_F(RasDiagnosticsMicrotest, InProgress_ClientWithoutDiagnosticsIsFalse) {
  LinkClient(MakeClient());
  EXPECT_FALSE(rasDiagnosticsInProgress());
}

TEST_F(RasDiagnosticsMicrotest, InProgress_ClientWithDiagnosticsIsTrue) {
  auto* client = MakeClient();
  struct rasDiagnosticsContext ctx{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, nullptr));
  LinkClient(client);
  EXPECT_TRUE(rasDiagnosticsInProgress());
}

TEST_F(RasDiagnosticsMicrotest, CancelTarget_NullTargetIsNoOp) {
  int target;
  auto* client = MakeClient();
  auto reporter = MakeRecordingReporter();
  reporter.target = &target;
  struct rasDiagnosticsContext ctx{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, &reporter));
  LinkClient(client);
  rasDiagnosticsCancelTarget(nullptr);
  EXPECT_EQ(RecordingEmit, client->diagnostics->reporter.emit);
}

TEST_F(RasDiagnosticsMicrotest, CancelTarget_MatchingTargetSwapsToNoopReporter) {
  int target;
  auto* client = MakeClient();
  auto reporter = MakeRecordingReporter();
  reporter.target = &target;
  struct rasDiagnosticsContext ctx{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, &reporter));
  LinkClient(client);
  rasDiagnosticsCancelTarget(&target);
  EXPECT_NE(RecordingEmit, client->diagnostics->reporter.emit);
  // The noop reporter must not crash or touch state when invoked.
  EXPECT_EQ(ncclSuccess, client->diagnostics->reporter.emit(client->diagnostics->reporter.target, "ignored"));
  EXPECT_EQ(0, g_emitCalls);
}

TEST_F(RasDiagnosticsMicrotest, CancelTarget_NonMatchingTargetLeavesReporterAlone) {
  int target, other;
  auto* client = MakeClient();
  auto reporter = MakeRecordingReporter();
  reporter.target = &target;
  struct rasDiagnosticsContext ctx{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, &reporter));
  LinkClient(client);
  rasDiagnosticsCancelTarget(&other);
  EXPECT_EQ(RecordingEmit, client->diagnostics->reporter.emit);
}

// ===========================================================================
// rasCollDiagInit / rasDiagnosticsCollectLocalPeerPayload (static, indirect)
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, CollDiagInit_UnscopedRequestPropagatesNoFilter) {
  struct rasCollRequest req{};
  req.diag.hasCommFilter = false;
  char* data = nullptr;
  int nData = 0;
  size_t reqLen = 0;
  struct rasCollRequest* pReq = &req;
  ASSERT_EQ(ncclSuccess, rasCollDiagInit(&pReq, &reqLen, &data, &nData));
  ASSERT_NE(nullptr, data);
  auto* peerHeader = reinterpret_cast<struct rasDiagnosticsPeerPayloadHeader*>(data);
  EXPECT_EQ(0, peerHeader->nChecks);  // No hooks produced records by default.
  EXPECT_EQ((int)sizeof(*peerHeader), peerHeader->payloadBytes);
  free(data);
}

TEST_F(RasDiagnosticsMicrotest, CollDiagInit_ScopedRequestPassesFilterToChecks) {
  struct rasCollRequest req{};
  req.diag.hasCommFilter = true;
  req.diag.commFilter.commHash = 0x99;
  g_checkHooks[0].collectLocalData.nRecords = 1;
  g_checkHooks[0].collectLocalData.recordStride = 4;
  g_checkHooks[0].collectLocalData.recordsBytes = 4;
  g_checkHooks[0].collectLocalData.records = static_cast<char*>(calloc(1, 4));
  char* data = nullptr;
  int nData = 0;
  size_t reqLen = 0;
  struct rasCollRequest* pReq = &req;
  ASSERT_EQ(ncclSuccess, rasCollDiagInit(&pReq, &reqLen, &data, &nData));
  ASSERT_EQ(1, g_checkHooks[0].collectLocalCalls);
  auto* peerHeader = reinterpret_cast<struct rasDiagnosticsPeerPayloadHeader*>(data);
  EXPECT_EQ(1, peerHeader->nChecks);
  free(data);
}

TEST_F(RasDiagnosticsMicrotest, CollDiagInit_CheckWithZeroRecordsIsOmitted) {
  struct rasCollRequest req{};
  // All hooks default to zero records.
  char* data = nullptr;
  int nData = 0;
  size_t reqLen = 0;
  struct rasCollRequest* pReq = &req;
  ASSERT_EQ(ncclSuccess, rasCollDiagInit(&pReq, &reqLen, &data, &nData));
  for (auto& h : g_checkHooks) EXPECT_EQ(1, h.collectLocalCalls);
  auto* peerHeader = reinterpret_cast<struct rasDiagnosticsPeerPayloadHeader*>(data);
  EXPECT_EQ(0, peerHeader->nChecks);
  free(data);
}

TEST_F(RasDiagnosticsMicrotest, CollDiagInit_CheckFailurePropagatesErrorAndFreesPayload) {
  struct rasCollRequest req{};
  g_checkHooks[2].collectLocalResult = ncclSystemError;
  char* data = nullptr;
  int nData = 0;
  size_t reqLen = 0;
  struct rasCollRequest* pReq = &req;
  EXPECT_EQ(ncclSystemError, rasCollDiagInit(&pReq, &reqLen, &data, &nData));
}

// ===========================================================================
// rasCollDiagMerge
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, CollDiagMerge_NullArgsReturnInternalError) {
  struct rasCollective coll{};
  struct rasMsg msg{};
  EXPECT_EQ(ncclInternalError, rasCollDiagMerge(nullptr, &msg));
  EXPECT_EQ(ncclInternalError, rasCollDiagMerge(&coll, nullptr));
}

TEST_F(RasDiagnosticsMicrotest, CollDiagMerge_NegativeSizesReturnInternalError) {
  struct rasCollective coll{};
  coll.nData = -1;
  struct rasMsg msg{};
  EXPECT_EQ(ncclInternalError, rasCollDiagMerge(&coll, &msg));
}

TEST_F(RasDiagnosticsMicrotest, CollDiagMerge_ZeroIncomingDataIsNoOp) {
  struct rasCollective coll{};
  struct rasMsg msg{};
  msg.collResp.nData = 0;
  EXPECT_EQ(ncclSuccess, rasCollDiagMerge(&coll, &msg));
  EXPECT_EQ(0, coll.nData);
}

TEST_F(RasDiagnosticsMicrotest, CollDiagMerge_RejectsPeerCountAndPayloadSizeOverflow) {
  struct rasCollective coll{};
  struct rasMsg msg{};
  msg.collResp.nData = 1;
  msg.collResp.nPeers = INT_MAX;
  EXPECT_EQ(ncclInternalError, rasCollDiagMerge(&coll, &msg));

  msg.collResp.nPeers = 0;
  coll.nData = INT_MAX;
  EXPECT_EQ(ncclInternalError, rasCollDiagMerge(&coll, &msg));
}

TEST_F(RasDiagnosticsMicrotest, CollDiagMerge_AppendsIncomingBytes) {
  struct rasCollective coll{};
  coll.data = static_cast<char*>(calloc(1, 4));
  memcpy(coll.data, "abcd", 4);
  coll.nData = 4;

  std::vector<char> extra = {'w', 'x', 'y', 'z'};
  int msgLen = static_cast<int>(rasMsgLength(RAS_MSG_COLLRESP));
  int dataOffset = msgLen;
  ALIGN_SIZE(dataOffset, alignof(int64_t));
  msgLen = dataOffset + static_cast<int>(extra.size());
  std::vector<char> buf(msgLen, 0);
  auto* msg = reinterpret_cast<struct rasMsg*>(buf.data());
  msg->collResp.nData = static_cast<int>(extra.size());
  msg->collResp.nPeers = 0;
  memcpy(buf.data() + dataOffset, extra.data(), extra.size());

  ASSERT_EQ(ncclSuccess, rasCollDiagMerge(&coll, msg));
  EXPECT_EQ(8, coll.nData);
  EXPECT_EQ(0, memcmp(coll.data, "abcdwxyz", 8));
  free(coll.data);
}

// ===========================================================================
// rasDiagnosticsResume (drives the static rasDiagnosticsSummarizePeerPayloads)
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, Resume_InvalidClientStateReturnsInternalError) {
  auto* client = MakeClient();
  EXPECT_EQ(ncclInternalError, rasDiagnosticsResume(client));  // No diagnostics, no coll.
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Resume_EmptyGatheredDataStillEmitsHeaderAndFooter) {
  auto* client = MakeClient();
  auto reporter = MakeRecordingReporter();
  struct rasDiagnosticsContext ctx{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, &reporter));
  client->coll = static_cast<struct rasCollective*>(calloc(1, sizeof(struct rasCollective)));
  client->coll->nPeers = 3;

  ASSERT_EQ(ncclSuccess, rasDiagnosticsResume(client));
  ASSERT_GE(g_emitCalls, 2);
  EXPECT_EQ("=== RAS Diagnostics ===", g_emittedLines.front());
  EXPECT_NE(std::string::npos, g_emittedLines.back().find("3 RAS peers"));
  EXPECT_EQ(1, g_finishCalls);
  EXPECT_EQ(ncclSuccess, g_lastFinishResult);
  EXPECT_EQ(1, g_collFreeCalls);
  EXPECT_EQ(nullptr, client->coll);
  EXPECT_EQ(nullptr, client->diagnostics);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Resume_InternalClientSkipsHeaderLine) {
  auto* client = MakeClient();
  client->internal = true;
  auto reporter = MakeRecordingReporter();
  struct rasDiagnosticsContext ctx{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, &reporter));
  client->coll = static_cast<struct rasCollective*>(calloc(1, sizeof(struct rasCollective)));

  ASSERT_EQ(ncclSuccess, rasDiagnosticsResume(client));
  for (auto& line : g_emittedLines) EXPECT_EQ(std::string::npos, line.find("=== RAS Diagnostics ==="));
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Resume_CommScopedReportsCommNRanks) {
  auto* client = MakeClient();
  auto reporter = MakeRecordingReporter();
  struct rasDiagnosticsContext ctx{};
  ctx.hasCommFilter = true;
  ctx.commNRanks = 7;
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, &reporter));
  client->coll = static_cast<struct rasCollective*>(calloc(1, sizeof(struct rasCollective)));
  client->coll->nPeers = 999;  // Should be ignored in favor of ctx.commNRanks.

  ASSERT_EQ(ncclSuccess, rasDiagnosticsResume(client));
  EXPECT_NE(std::string::npos, g_emittedLines.back().find("7 ranks"));
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Resume_SummarizeCalledForEveryCheckEvenWithoutRecords) {
  auto* client = MakeClient();
  auto reporter = MakeRecordingReporter();
  struct rasDiagnosticsContext ctx{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, &reporter));
  client->coll = static_cast<struct rasCollective*>(calloc(1, sizeof(struct rasCollective)));

  auto gathered = BuildGatheredData({{{RAS_DIAG_CHECK_ECC, 4, {1, 2, 3, 4}}}});
  client->coll->data = static_cast<char*>(calloc(gathered.size(), 1));
  memcpy(client->coll->data, gathered.data(), gathered.size());
  client->coll->nData = static_cast<int>(gathered.size());

  ASSERT_EQ(ncclSuccess, rasDiagnosticsResume(client));
  for (int id = 0; id < RAS_DIAG_CHECK_COUNT; id++) EXPECT_EQ(1, g_checkHooks[id].summarizeCalls);
  EXPECT_EQ(4, g_checkHooks[2].lastSummarizeNData);
  ASSERT_EQ(4u, g_checkHooks[2].lastSummarizeData.size());
  EXPECT_EQ(0, memcmp(g_checkHooks[2].lastSummarizeData.data(), "\x01\x02\x03\x04", 4));
  // A check that no peer contributed to is still summarized, with an empty payload.
  EXPECT_EQ(1, g_checkHooks[0].summarizeCalls);
  EXPECT_EQ(0, g_checkHooks[0].lastSummarizeNData);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Resume_MultiplePeersAccumulateSameCheck) {
  auto* client = MakeClient();
  auto reporter = MakeRecordingReporter();
  struct rasDiagnosticsContext ctx{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, &reporter));
  client->coll = static_cast<struct rasCollective*>(calloc(1, sizeof(struct rasCollective)));

  auto gathered = BuildGatheredData({{{RAS_DIAG_CHECK_GPU_MODEL, 4, {1, 1, 1, 1}}},
                                     {{RAS_DIAG_CHECK_GPU_MODEL, 4, {2, 2, 2, 2}}}});
  client->coll->data = static_cast<char*>(calloc(gathered.size(), 1));
  memcpy(client->coll->data, gathered.data(), gathered.size());
  client->coll->nData = static_cast<int>(gathered.size());

  ASSERT_EQ(ncclSuccess, rasDiagnosticsResume(client));
  EXPECT_EQ(8, g_checkHooks[0].lastSummarizeNData);
  ASSERT_EQ(8u, g_checkHooks[0].lastSummarizeData.size());
  const char expected[8] = {1, 1, 1, 1, 2, 2, 2, 2};
  EXPECT_EQ(0, memcmp(g_checkHooks[0].lastSummarizeData.data(), expected, 8));
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Resume_TruncatedPeerHeaderReturnsErrorButStillFinishes) {
  auto* client = MakeClient();
  auto reporter = MakeRecordingReporter();
  struct rasDiagnosticsContext ctx{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, &reporter));
  client->coll = static_cast<struct rasCollective*>(calloc(1, sizeof(struct rasCollective)));
  client->coll->data = static_cast<char*>(calloc(1, 2));
  client->coll->nData = 2;  // Smaller than sizeof(rasDiagnosticsPeerPayloadHeader).

  ASSERT_EQ(ncclInternalError, rasDiagnosticsResume(client));
  EXPECT_EQ(1, g_finishCalls);
  EXPECT_EQ(ncclInternalError, g_lastFinishResult);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Resume_UnknownCheckIdReturnsError) {
  auto* client = MakeClient();
  auto reporter = MakeRecordingReporter();
  struct rasDiagnosticsContext ctx{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, &reporter));
  client->coll = static_cast<struct rasCollective*>(calloc(1, sizeof(struct rasCollective)));

  auto gathered = BuildGatheredData({{{static_cast<rasDiagnosticsCheckId>(99), 4, {1, 2, 3, 4}}}});
  client->coll->data = static_cast<char*>(calloc(gathered.size(), 1));
  memcpy(client->coll->data, gathered.data(), gathered.size());
  client->coll->nData = static_cast<int>(gathered.size());

  EXPECT_EQ(ncclInternalError, rasDiagnosticsResume(client));
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, SummarizePeerPayloads_RejectsInvalidTopLevelArguments) {
  struct rasDiagnosticsContext ctx{};
  const char byte = 0;
  EXPECT_EQ(ncclInternalError, rasDiagnosticsSummarizePeerPayloads(nullptr, nullptr, &byte, 1));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsSummarizePeerPayloads(&ctx, nullptr, nullptr, 1));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsSummarizePeerPayloads(&ctx, nullptr, &byte, -1));
}

TEST_F(RasDiagnosticsMicrotest, SummarizePeerPayloads_RejectsMalformedPeerHeaders) {
  struct rasDiagnosticsContext ctx{};
  std::vector<char> payload(sizeof(struct rasDiagnosticsPeerPayloadHeader), 0);
  auto* header = reinterpret_cast<struct rasDiagnosticsPeerPayloadHeader*>(payload.data());

  header->nChecks = -1;
  header->payloadBytes = static_cast<int>(payload.size());
  EXPECT_EQ(ncclInternalError,
            rasDiagnosticsSummarizePeerPayloads(&ctx, nullptr, payload.data(), static_cast<int>(payload.size())));

  header->nChecks = 0;
  header->payloadBytes = static_cast<int>(sizeof(*header)) - 1;
  EXPECT_EQ(ncclInternalError,
            rasDiagnosticsSummarizePeerPayloads(&ctx, nullptr, payload.data(), static_cast<int>(payload.size())));

  header->payloadBytes = static_cast<int>(payload.size()) + 1;
  EXPECT_EQ(ncclInternalError,
            rasDiagnosticsSummarizePeerPayloads(&ctx, nullptr, payload.data(), static_cast<int>(payload.size())));
}

TEST_F(RasDiagnosticsMicrotest, SummarizePeerPayloads_RejectsTruncatedCheckHeaderAndTrailingBytes) {
  struct rasDiagnosticsContext ctx{};
  std::vector<char> payload(sizeof(struct rasDiagnosticsPeerPayloadHeader), 0);
  auto* peerHeader = reinterpret_cast<struct rasDiagnosticsPeerPayloadHeader*>(payload.data());
  peerHeader->nChecks = 1;
  peerHeader->payloadBytes = static_cast<int>(payload.size());
  EXPECT_EQ(ncclInternalError,
            rasDiagnosticsSummarizePeerPayloads(&ctx, nullptr, payload.data(), static_cast<int>(payload.size())));

  payload.push_back(0);
  peerHeader = reinterpret_cast<struct rasDiagnosticsPeerPayloadHeader*>(payload.data());
  peerHeader->nChecks = 0;
  peerHeader->payloadBytes = static_cast<int>(payload.size());
  EXPECT_EQ(ncclInternalError,
            rasDiagnosticsSummarizePeerPayloads(&ctx, nullptr, payload.data(), static_cast<int>(payload.size())));
}

TEST_F(RasDiagnosticsMicrotest, SummarizePeerPayloads_RejectsMalformedCheckMetadata) {
  struct rasDiagnosticsContext ctx{};
  auto expectRejected = [&](auto mutate) {
    auto payload = BuildPeerPayload({{RAS_DIAG_CHECK_GPU_MODEL, 4, {1, 2, 3, 4}}});
    mutate(*FirstCheckHeader(payload));
    EXPECT_EQ(ncclInternalError,
              rasDiagnosticsSummarizePeerPayloads(&ctx, nullptr, payload.data(), static_cast<int>(payload.size())));
  };

  expectRejected([](auto& header) { header.recordStride = 0; });
  expectRejected([](auto& header) { header.nRecords = -1; });
  expectRejected([](auto& header) { header.payloadBytes = -1; });
  expectRejected([](auto& header) { header.payloadBytes = 5; });
  expectRejected([](auto& header) {
    header.recordStride = 2;
    header.nRecords = INT_MAX;
  });
  expectRejected([](auto& header) { header.nRecords = 2; });
}

TEST_F(RasDiagnosticsMicrotest, SummarizePeerPayloads_RejectsStrideChangesAcrossPeers) {
  struct rasDiagnosticsContext ctx{};
  auto gathered = BuildGatheredData({{{RAS_DIAG_CHECK_GPU_MODEL, 4, {1, 2, 3, 4}}},
                                     {{RAS_DIAG_CHECK_GPU_MODEL, 2, {5, 6}}}});
  EXPECT_EQ(ncclInternalError,
            rasDiagnosticsSummarizePeerPayloads(&ctx, nullptr, gathered.data(), static_cast<int>(gathered.size())));
}

TEST_F(RasDiagnosticsMicrotest, Resume_ReporterFinishFailureIsLoggedButResultStillReturned) {
  auto* client = MakeClient();
  auto reporter = MakeRecordingReporter();
  g_finishResult = ncclSystemError;
  struct rasDiagnosticsContext ctx{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, &reporter));
  client->coll = static_cast<struct rasCollective*>(calloc(1, sizeof(struct rasCollective)));

  EXPECT_EQ(ncclSuccess, rasDiagnosticsResume(client));  // finish()'s own result doesn't override ret.
  EXPECT_EQ(1, g_finishCalls);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Resume_NullReporterFinishDoesNotCrash) {
  auto* client = MakeClient();
  struct rasDiagnosticsContext ctx{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, nullptr));  // Default reporter has finish==nullptr.
  client->coll = static_cast<struct rasCollective*>(calloc(1, sizeof(struct rasCollective)));

  EXPECT_EQ(ncclSuccess, rasDiagnosticsResume(client));
  free(client);
}

// ===========================================================================
// rasDiagnosticsStart
// ===========================================================================

TEST_F(RasDiagnosticsMicrotest, Start_NullClientReturnsInternalError) {
  EXPECT_EQ(ncclInternalError, rasDiagnosticsStart(nullptr));
}

TEST_F(RasDiagnosticsMicrotest, Start_MissingDiagnosticsStateReturnsInternalError) {
  auto* client = MakeClient();
  EXPECT_EQ(ncclInternalError, rasDiagnosticsStart(client));
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Start_PostsCollectiveAndAdvancesStateWhenIncomplete) {
  auto* client = MakeClient();
  struct rasDiagnosticsContext ctx{};
  ctx.hasCommFilter = true;
  ctx.commFilter.commHash = 0x77;
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, nullptr));
  client->timeout = 12345;
  g_netSendCollReqAllDone = false;
  auto* fakeColl = static_cast<struct rasCollective*>(calloc(1, sizeof(struct rasCollective)));
  g_netSendCollReqCollToAssign = fakeColl;

  EXPECT_EQ(ncclInProgress, rasDiagnosticsStart(client));
  EXPECT_EQ(1, g_netSendCollReqCalls);
  EXPECT_EQ(RAS_COLL_DIAG, g_lastSentReq.type);
  EXPECT_EQ(12345, g_lastSentReq.timeout);
  EXPECT_TRUE(g_lastSentReq.diag.hasCommFilter);
  EXPECT_EQ(0x77u, g_lastSentReq.diag.commFilter.commHash);
  EXPECT_EQ(RAS_CLIENT_DIAG_FINI, client->status);
  EXPECT_EQ(fakeColl, client->coll);
  free(client->coll);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Start_ReturnsSuccessWhenAlreadyAllDone) {
  auto* client = MakeClient();
  struct rasDiagnosticsContext ctx{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, nullptr));
  g_netSendCollReqAllDone = true;

  EXPECT_EQ(ncclSuccess, rasDiagnosticsStart(client));
  EXPECT_EQ(RAS_CLIENT_DIAG_FINI, client->status);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Start_SendFailureFreesCollAndCleansUpDiagnostics) {
  auto* client = MakeClient();
  struct rasDiagnosticsContext ctx{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, nullptr));
  g_netSendCollReqResult = ncclSystemError;

  EXPECT_EQ(ncclSystemError, rasDiagnosticsStart(client));
  EXPECT_EQ(nullptr, client->coll);
  EXPECT_EQ(nullptr, client->diagnostics);
  free(client);
}

TEST_F(RasDiagnosticsMicrotest, Start_SendFailureFreesPartiallyCreatedCollective) {
  auto* client = MakeClient();
  struct rasDiagnosticsContext ctx{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsClientInit(client, &ctx, nullptr));
  client->coll = static_cast<struct rasCollective*>(calloc(1, sizeof(struct rasCollective)));
  g_netSendCollReqResult = ncclSystemError;

  EXPECT_EQ(ncclSystemError, rasDiagnosticsStart(client));
  EXPECT_EQ(1, g_collFreeCalls);
  EXPECT_EQ(nullptr, client->coll);
  EXPECT_EQ(nullptr, client->diagnostics);
  free(client);
}

#undef rasDiagnosticsNcclEnvSummarize
#undef rasDiagnosticsNcclEnvCollectLocal
#undef rasDiagnosticsNvLinkSummarize
#undef rasDiagnosticsNvLinkCollectLocal
#undef rasDiagnosticsEccSummarize
#undef rasDiagnosticsEccCollectLocal
#undef rasDiagnosticsCudaDriverVersionSummarize
#undef rasDiagnosticsCudaDriverVersionCollectLocal
#undef rasDiagnosticsGpuModelSummarize
#undef rasDiagnosticsGpuModelCollectLocal
#undef rasNetSendCollReq
#undef rasCollReqInit
#undef rasCollFree
#undef rasClientsHead
