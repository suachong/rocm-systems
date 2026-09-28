/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ras/diagnostics_env.cc.

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "comm.h"
#include "ras/diagnostics_checks_common.h"

#include DIAGNOSTICS_ENV_CC_PATH

namespace {

int g_emitCalls;
std::vector<std::string> g_emittedLines;

ncclResult_t RecordingEmit(void* /*target*/, const char* line) {
  ++g_emitCalls;
  g_emittedLines.emplace_back(line);
  return ncclSuccess;
}

rasDiagnosticsReporter MakeRecordingReporter() {
  rasDiagnosticsReporter reporter{};
  reporter.emit = RecordingEmit;
  return reporter;
}

bool AnyLineContains(const std::vector<std::string>& lines, const std::string& needle) {
  for (auto& l : lines)
    if (l.find(needle) != std::string::npos) return true;
  return false;
}

struct FakeComm {
  std::unique_ptr<ncclComm> comm{new ncclComm{}};
  std::vector<ncclPeerInfo> peerInfos;

  FakeComm(uint64_t commHash, int rank, int nRanks) {
    comm->commHash = commHash;
    comm->rank = rank;
    comm->nRanks = nRanks;
    comm->peerInfoValid = true;
    comm->cudaDev = 0;
    comm->nvmlDev = 0;
    comm->busId = 0;
    comm->localRank = 0;
    comm->localRanks = 1;
    peerInfos.assign(1, ncclPeerInfo{});
    peerInfos[0].hostHash = commHash + 1;
    peerInfos[0].pidHash = commHash + 2;
    comm->peerInfo = peerInfos.data();
  }
  ncclComm* get() { return comm.get(); }
};

void InstallNcclComms(std::vector<ncclComm*> comms) {
  free(ncclComms);
  nNcclComms = static_cast<int>(comms.size());
  ncclComms = static_cast<ncclComm**>(calloc(comms.size() ? comms.size() : 1, sizeof(*ncclComms)));
  for (size_t i = 0; i < comms.size(); i++) ncclComms[i] = comms[i];
}

// --------- Fake `environ` plumbing ---------

char** g_realEnviron;
std::vector<std::string> g_envStorage;
std::vector<char*> g_envPointers;

void SetFakeEnviron(const std::vector<std::string>& entries) {
  g_envStorage = entries;
  g_envPointers.clear();
  for (auto& s : g_envStorage) g_envPointers.push_back(const_cast<char*>(s.c_str()));
  g_envPointers.push_back(nullptr);
  environ = g_envPointers.data();
}

void ResetWholeFileSeams() {
  g_emitCalls = 0;
  g_emittedLines.clear();
  free(ncclComms);
  ncclComms = nullptr;
  nNcclComms = 0;
  SetFakeEnviron({});
}

class RasDiagnosticsEnvMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_realEnviron = environ;
    ResetWholeFileSeams();
  }
  void TearDown() override {
    ResetWholeFileSeams();
    environ = g_realEnviron;
  }
};

}  // namespace

// ===========================================================================
// rasDiagnosticsNcclEnvCollectLocal (via the real rasDiagnosticsCollectLocalRecords)
// ===========================================================================

TEST_F(RasDiagnosticsEnvMicrotest, CollectLocal_NoMatchingCommsReturnsEmpty) {
  struct rasDiagnosticsContext ctx{};
  struct rasDiagnosticsLocalData data{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNcclEnvCollectLocal(&ctx, &data));
  EXPECT_EQ(0, data.nRecords);
}

TEST_F(RasDiagnosticsEnvMicrotest, CollectLocal_FiltersNonNcclPrefixedVars) {
  FakeComm fc(0x1000, 0, 1);
  InstallNcclComms({fc.get()});
  SetFakeEnviron({"PATH=/usr/bin", "NCCL_DEBUG=INFO", "HOME=/root"});

  struct rasDiagnosticsContext ctx{};
  struct rasDiagnosticsLocalData data{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNcclEnvCollectLocal(&ctx, &data));
  ASSERT_EQ(1, data.nRecords);
  const auto* envData = reinterpret_cast<const struct rasDiagnosticsNcclEnvData*>(
    data.records + sizeof(struct rasDiagnosticsRankHeader));
  EXPECT_EQ(0, envData->truncated);
  std::string blob(envData->data, envData->bytesUsed);
  EXPECT_NE(std::string::npos, blob.find("NCCL_DEBUG=INFO"));
  EXPECT_EQ(std::string::npos, blob.find("PATH="));
  free(data.records);
}

TEST_F(RasDiagnosticsEnvMicrotest, CollectLocal_NoNcclVarsProducesEmptyPayload) {
  FakeComm fc(0x1000, 0, 1);
  InstallNcclComms({fc.get()});
  SetFakeEnviron({"PATH=/usr/bin"});

  struct rasDiagnosticsContext ctx{};
  struct rasDiagnosticsLocalData data{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNcclEnvCollectLocal(&ctx, &data));
  ASSERT_EQ(1, data.nRecords);
  const auto* envData = reinterpret_cast<const struct rasDiagnosticsNcclEnvData*>(
    data.records + sizeof(struct rasDiagnosticsRankHeader));
  EXPECT_EQ(0, envData->bytesUsed);
  free(data.records);
}

TEST_F(RasDiagnosticsEnvMicrotest, CollectLocal_ExceedingByteBudgetSetsTruncated) {
  FakeComm fc(0x1000, 0, 1);
  InstallNcclComms({fc.get()});
  std::vector<std::string> entries;
  // Each ~1040 bytes; ~17 of them exceed the 16384-byte budget.
  for (int i = 0; i < 17; i++) entries.push_back("NCCL_VAR" + std::to_string(i) + "=" + std::string(1024, 'a'));
  SetFakeEnviron(entries);

  struct rasDiagnosticsContext ctx{};
  struct rasDiagnosticsLocalData data{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNcclEnvCollectLocal(&ctx, &data));
  ASSERT_EQ(1, data.nRecords);
  const auto* envData = reinterpret_cast<const struct rasDiagnosticsNcclEnvData*>(
    data.records + sizeof(struct rasDiagnosticsRankHeader));
  EXPECT_EQ(1, envData->truncated);
  free(data.records);
}

TEST_F(RasDiagnosticsEnvMicrotest, CollectLocal_ExactFitAtTheByteBudgetIsNotTruncated) {
  FakeComm fc(0x1000, 0, 1);
  InstallNcclComms({fc.get()});
  // "NCCL_X=" (7) + value: total strlen == RAS_DIAG_ENV_BYTES - 1, so len (strlen+1,
  // including the NUL) exactly equals the remaining budget -- fits exactly, at the
  // boundary between "fits" and "doesn't fit".
  std::string value(RAS_DIAG_ENV_BYTES - 1 - 7, 'a');
  SetFakeEnviron({"NCCL_X=" + value});

  struct rasDiagnosticsContext ctx{};
  struct rasDiagnosticsLocalData data{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNcclEnvCollectLocal(&ctx, &data));
  ASSERT_EQ(1, data.nRecords);
  const auto* envData = reinterpret_cast<const struct rasDiagnosticsNcclEnvData*>(
    data.records + sizeof(struct rasDiagnosticsRankHeader));
  EXPECT_EQ(0, envData->truncated);
  EXPECT_EQ(RAS_DIAG_ENV_BYTES, envData->bytesUsed);
  free(data.records);
}

// ===========================================================================
// rasDiagnosticsNcclEnvSummarize: argument validation
// ===========================================================================

TEST_F(RasDiagnosticsEnvMicrotest, Summarize_NullReporterReturnsInternalError) {
  EXPECT_EQ(ncclInternalError, rasDiagnosticsNcclEnvSummarize(nullptr, nullptr, nullptr, 0));
}

TEST_F(RasDiagnosticsEnvMicrotest, Summarize_ReporterWithNullEmitReturnsInternalError) {
  struct rasDiagnosticsReporter reporter{};
  EXPECT_EQ(ncclInternalError, rasDiagnosticsNcclEnvSummarize(nullptr, &reporter, nullptr, 0));
}

TEST_F(RasDiagnosticsEnvMicrotest, Summarize_ZeroDataIsNoOp) {
  auto reporter = MakeRecordingReporter();
  EXPECT_EQ(ncclSuccess, rasDiagnosticsNcclEnvSummarize(nullptr, &reporter, nullptr, 0));
  EXPECT_EQ(0, g_emitCalls);
}

TEST_F(RasDiagnosticsEnvMicrotest, Summarize_NullDataWithNonzeroSizeReturnsInternalError) {
  auto reporter = MakeRecordingReporter();
  EXPECT_EQ(ncclInternalError, rasDiagnosticsNcclEnvSummarize(nullptr, &reporter, nullptr, 8));
}

TEST_F(RasDiagnosticsEnvMicrotest, Summarize_SizeNotMultipleOfStrideReturnsInternalError) {
  auto reporter = MakeRecordingReporter();
  char buf[4] = {};
  EXPECT_EQ(ncclInternalError, rasDiagnosticsNcclEnvSummarize(nullptr, &reporter, buf, 1));
}

// ===========================================================================
// rasDiagnosticsNcclEnvSummarize: end-to-end via CollectLocal-produced records
// ===========================================================================

namespace {

// Runs CollectLocal for the currently installed ncclComms/environ, then feeds
// the result straight into Summarize -- exercising the exact production
// pairing (one rank's real gathered record, not a hand-built stand-in).
ncclResult_t CollectThenSummarize(struct rasDiagnosticsReporter* reporter) {
  struct rasDiagnosticsContext ctx{};
  struct rasDiagnosticsLocalData data{};
  ncclResult_t ret = rasDiagnosticsNcclEnvCollectLocal(&ctx, &data);
  if (ret != ncclSuccess) return ret;
  ret = rasDiagnosticsNcclEnvSummarize(&ctx, reporter, data.records, data.recordsBytes);
  free(data.records);
  return ret;
}

}  // namespace

TEST_F(RasDiagnosticsEnvMicrotest, Summarize_SingleRankConsistentReport) {
  FakeComm fc(0x2000, 0, 1);
  InstallNcclComms({fc.get()});
  SetFakeEnviron({"NCCL_DEBUG=INFO"});
  auto reporter = MakeRecordingReporter();

  ASSERT_EQ(ncclSuccess, CollectThenSummarize(&reporter));
  EXPECT_TRUE(AnyLineContains(g_emittedLines, "NCCL_* env vars consistent across 1 ranks"));
}

TEST_F(RasDiagnosticsEnvMicrotest, Summarize_IncompleteGroupReportsIncomplete) {
  // commNRanks says 2, but only rank 0 is present in ncclComms -- an incomplete gather.
  FakeComm fc(0x3000, 0, 2);
  InstallNcclComms({fc.get()});
  SetFakeEnviron({});
  auto reporter = MakeRecordingReporter();

  ASSERT_EQ(ncclSuccess, CollectThenSummarize(&reporter));
  EXPECT_TRUE(AnyLineContains(g_emittedLines, "diagnostics incomplete, gathered 1/2 ranks"));
}

TEST_F(RasDiagnosticsEnvMicrotest, Summarize_TwoRanksMismatchedValueReportsMismatch) {
  // Two distinct ncclComm entries, same commHash/hostHash/pidHash pairing style but
  // different rank -- built by hand below since FakeComm's per-object environ can't
  // differ per rank (only one process-wide `environ`), so we hand-build two records
  // directly instead of going through CollectLocal twice.
  const size_t stride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsNcclEnvData));
  std::vector<char> buf(stride * 2, 0);

  auto fillRecord = [&](int rank, const char* value) {
    auto* rank_hdr = reinterpret_cast<struct rasDiagnosticsRankHeader*>(buf.data() + rank * stride);
    rank_hdr->commId.commHash = 0x4000;
    rank_hdr->commId.hostHash = 0x4001;
    rank_hdr->commId.pidHash = 0x4002;
    rank_hdr->commRank = rank;
    rank_hdr->commNRanks = 2;
    auto* envData = reinterpret_cast<struct rasDiagnosticsNcclEnvData*>(buf.data() + rank * stride +
                                                                        sizeof(struct rasDiagnosticsRankHeader));
    std::string entry = std::string("NCCL_DEBUG=") + value;
    memcpy(envData->data, entry.c_str(), entry.size() + 1);
    envData->bytesUsed = static_cast<uint16_t>(entry.size() + 1);
    envData->truncated = 0;
  };
  fillRecord(0, "INFO");
  fillRecord(1, "WARN");

  auto reporter = MakeRecordingReporter();
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNcclEnvSummarize(nullptr, &reporter, buf.data(), static_cast<int>(buf.size())));
  EXPECT_TRUE(AnyLineContains(g_emittedLines, "mismatch across 2 ranks"));
  EXPECT_TRUE(AnyLineContains(g_emittedLines, "NCCL_DEBUG=INFO on rank(s) {0}"));
  EXPECT_TRUE(AnyLineContains(g_emittedLines, "NCCL_DEBUG=WARN on rank(s) {1}"));
  EXPECT_TRUE(AnyLineContains(g_emittedLines, "1 NCCL_* env var(s) differ"));
}

TEST_F(RasDiagnosticsEnvMicrotest, Summarize_ValueSetOnOneRankUnsetOnAnotherIsAMismatch) {
  const size_t stride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsNcclEnvData));
  std::vector<char> buf(stride * 2, 0);

  auto* r0 = reinterpret_cast<struct rasDiagnosticsRankHeader*>(buf.data());
  r0->commId.commHash = 0x5000;
  r0->commRank = 0;
  r0->commNRanks = 2;
  auto* env0 =
    reinterpret_cast<struct rasDiagnosticsNcclEnvData*>(buf.data() + sizeof(struct rasDiagnosticsRankHeader));
  const char* entry0 = "NCCL_FOO=bar";
  memcpy(env0->data, entry0, strlen(entry0) + 1);
  env0->bytesUsed = static_cast<uint16_t>(strlen(entry0) + 1);

  auto* r1 = reinterpret_cast<struct rasDiagnosticsRankHeader*>(buf.data() + stride);
  r1->commId.commHash = 0x5000;
  r1->commRank = 1;
  r1->commNRanks = 2;
  auto* env1 = reinterpret_cast<struct rasDiagnosticsNcclEnvData*>(buf.data() + stride +
                                                                   sizeof(struct rasDiagnosticsRankHeader));
  env1->bytesUsed = 0;  // NCCL_FOO not set on rank 1.

  auto reporter = MakeRecordingReporter();
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNcclEnvSummarize(nullptr, &reporter, buf.data(), static_cast<int>(buf.size())));
  EXPECT_TRUE(AnyLineContains(g_emittedLines, "NCCL_FOO=(unset) on rank(s) {1}"));
  EXPECT_TRUE(AnyLineContains(g_emittedLines, "NCCL_FOO=bar on rank(s) {0}"));
}

TEST_F(RasDiagnosticsEnvMicrotest, Summarize_TruncatedRankReportsTruncationWarning) {
  const size_t stride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsNcclEnvData));
  std::vector<char> buf(stride, 0);
  auto* rank_hdr = reinterpret_cast<struct rasDiagnosticsRankHeader*>(buf.data());
  rank_hdr->commId.commHash = 0x6000;
  rank_hdr->commRank = 0;
  rank_hdr->commNRanks = 1;
  auto* envData =
    reinterpret_cast<struct rasDiagnosticsNcclEnvData*>(buf.data() + sizeof(struct rasDiagnosticsRankHeader));
  envData->bytesUsed = 0;
  envData->truncated = 1;

  auto reporter = MakeRecordingReporter();
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNcclEnvSummarize(nullptr, &reporter, buf.data(), static_cast<int>(buf.size())));
  EXPECT_TRUE(AnyLineContains(g_emittedLines, "1 rank(s) had >16384 bytes"));
}

TEST_F(RasDiagnosticsEnvMicrotest, Summarize_MalformedEnvPayloadReturnsInternalError) {
  const size_t stride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsNcclEnvData));
  std::vector<char> buf(stride, 0);
  auto* rank_hdr = reinterpret_cast<struct rasDiagnosticsRankHeader*>(buf.data());
  rank_hdr->commId.commHash = 0x7000;
  rank_hdr->commRank = 0;
  rank_hdr->commNRanks = 1;
  auto* envData =
    reinterpret_cast<struct rasDiagnosticsNcclEnvData*>(buf.data() + sizeof(struct rasDiagnosticsRankHeader));
  envData->bytesUsed = 5;
  envData->data[4] = 'x';  // Not NUL-terminated at the claimed end -- malformed.

  auto reporter = MakeRecordingReporter();
  EXPECT_EQ(ncclInternalError,
            rasDiagnosticsNcclEnvSummarize(nullptr, &reporter, buf.data(), static_cast<int>(buf.size())));
}

TEST_F(RasDiagnosticsEnvMicrotest, Summarize_TwoDistinctCommsAreReportedSeparately) {
  const size_t stride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsNcclEnvData));
  std::vector<char> buf(stride * 2, 0);

  auto fillSolo = [&](int idx, uint64_t commHash) {
    auto* rank_hdr = reinterpret_cast<struct rasDiagnosticsRankHeader*>(buf.data() + idx * stride);
    rank_hdr->commId.commHash = commHash;
    rank_hdr->commRank = 0;
    rank_hdr->commNRanks = 1;
    auto* envData = reinterpret_cast<struct rasDiagnosticsNcclEnvData*>(buf.data() + idx * stride +
                                                                        sizeof(struct rasDiagnosticsRankHeader));
    envData->bytesUsed = 0;
  };
  fillSolo(0, 0x8000);
  fillSolo(1, 0x9000);

  auto reporter = MakeRecordingReporter();
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNcclEnvSummarize(nullptr, &reporter, buf.data(), static_cast<int>(buf.size())));
  int consistentCount = 0;
  for (auto& l : g_emittedLines) {
    if (l.find("consistent across 1 ranks") != std::string::npos) consistentCount++;
  }
  EXPECT_EQ(2, consistentCount);
}

TEST_F(RasDiagnosticsEnvMicrotest, Summarize_LongKeyAndValueAreTruncatedWithEllipsis) {
  const size_t stride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsNcclEnvData));
  std::vector<char> buf(stride * 2, 0);

  std::string longKey = "NCCL_" + std::string(300, 'K');
  std::string longVal(300, 'V');
  std::string entry0 = longKey + "=" + longVal;
  std::string entry1 = longKey + "=" + std::string(300, 'W');  // Different value -> mismatch.

  auto fillRecord = [&](int rank, const std::string& entry) {
    auto* rank_hdr = reinterpret_cast<struct rasDiagnosticsRankHeader*>(buf.data() + rank * stride);
    rank_hdr->commId.commHash = 0xA000;
    rank_hdr->commRank = rank;
    rank_hdr->commNRanks = 2;
    auto* envData = reinterpret_cast<struct rasDiagnosticsNcclEnvData*>(buf.data() + rank * stride +
                                                                        sizeof(struct rasDiagnosticsRankHeader));
    memcpy(envData->data, entry.c_str(), entry.size() + 1);
    envData->bytesUsed = static_cast<uint16_t>(entry.size() + 1);
  };
  fillRecord(0, entry0);
  fillRecord(1, entry1);

  auto reporter = MakeRecordingReporter();
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNcclEnvSummarize(nullptr, &reporter, buf.data(), static_cast<int>(buf.size())));
  EXPECT_TRUE(AnyLineContains(g_emittedLines, "KKK..."));
  EXPECT_TRUE(AnyLineContains(g_emittedLines, "VVV..."));
}

TEST_F(RasDiagnosticsEnvMicrotest, Summarize_ControlCharactersAreSanitizedToQuestionMarks) {
  const size_t stride = rasDiagnosticsLocalRecordStride(sizeof(struct rasDiagnosticsNcclEnvData));
  std::vector<char> buf(stride * 2, 0);

  std::string entry0 = std::string("NCCL_X=a\x01" "b");
  std::string entry1 = "NCCL_X=different";

  auto fillRecord = [&](int rank, const std::string& entry) {
    auto* rank_hdr = reinterpret_cast<struct rasDiagnosticsRankHeader*>(buf.data() + rank * stride);
    rank_hdr->commId.commHash = 0xB000;
    rank_hdr->commRank = rank;
    rank_hdr->commNRanks = 2;
    auto* envData = reinterpret_cast<struct rasDiagnosticsNcclEnvData*>(buf.data() + rank * stride +
                                                                        sizeof(struct rasDiagnosticsRankHeader));
    memcpy(envData->data, entry.c_str(), entry.size() + 1);
    envData->bytesUsed = static_cast<uint16_t>(entry.size() + 1);
  };
  fillRecord(0, entry0);
  fillRecord(1, entry1);

  auto reporter = MakeRecordingReporter();
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNcclEnvSummarize(nullptr, &reporter, buf.data(), static_cast<int>(buf.size())));
  EXPECT_TRUE(AnyLineContains(g_emittedLines, "NCCL_X=a?b"));
}
