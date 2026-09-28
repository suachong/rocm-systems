/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ras/diagnostics_gpu.cc.

#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "comm.h"
#include "cudawrap.h"
#include "nvmlwrap.h"
#include "param.h"
#include "ras/diagnostics_checks_common.h"

namespace {

int64_t g_eccThreshold = 0;
ncclResult_t g_driverResult = ncclSuccess;
int g_driverVersion = 0;

}  // namespace

ncclResult_t DiagnosticsGpuTestCudaDriverVersion(int* version);

#define ncclCudaDriverVersion DiagnosticsGpuTestCudaDriverVersion
#undef NCCL_PARAM
#define NCCL_PARAM(name, env, defaultValue) int64_t ncclParam##name() { return g_eccThreshold; }

#include DIAGNOSTICS_GPU_CC_PATH

#undef NCCL_PARAM
#undef ncclCudaDriverVersion

int ncclNvmlDeviceCount = 0;
ncclNvmlDeviceInfo ncclNvmlDevices[ncclNvmlMaxDevices]{};
ncclNvmlDevicePairInfo ncclNvmlDevicePairs[ncclNvmlMaxDevices][ncclNvmlMaxDevices]{};

namespace {

ncclResult_t g_deviceCountResult = ncclSuccess;
unsigned int g_deviceCount = 0;
ncclResult_t g_handleResult = ncclSuccess;
ncclResult_t g_nameResult = ncclSuccess;
std::string g_deviceName;
std::array<std::array<unsigned long long, 2>, 2> g_eccCounters{};
std::array<std::array<ncclResult_t, 2>, 2> g_eccResults{};
std::array<unsigned int, RAS_DIAG_NVLINK_MAX_LINKS> g_nvLinkValid{};
std::array<ncclResult_t, RAS_DIAG_NVLINK_MAX_LINKS> g_nvLinkCapabilityResults{};
std::array<nvmlEnableState_t, RAS_DIAG_NVLINK_MAX_LINKS> g_nvLinkStates{};
std::array<ncclResult_t, RAS_DIAG_NVLINK_MAX_LINKS> g_nvLinkStateResults{};

int DeviceIndex(nvmlDevice_t device) {
  return static_cast<int>(reinterpret_cast<uintptr_t>(device)) - 1;
}

int ErrorIndex(nvmlMemoryErrorType_t type) {
  return type == NVML_MEMORY_ERROR_TYPE_UNCORRECTED ? 1 : 0;
}

int LocationIndex(nvmlMemoryLocation_t location) {
  return location == NVML_MEMORY_LOCATION_DRAM ? 1 : 0;
}

}  // namespace

ncclResult_t DiagnosticsGpuTestCudaDriverVersion(int* version) {
  if (g_driverResult != ncclSuccess) return g_driverResult;
  *version = g_driverVersion;
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetCount(unsigned int* deviceCount) {
  if (g_deviceCountResult != ncclSuccess) return g_deviceCountResult;
  *deviceCount = g_deviceCount;
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) {
  if (g_handleResult != ncclSuccess) return g_handleResult;
  *device = reinterpret_cast<nvmlDevice_t>(static_cast<uintptr_t>(index + 1));
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetName(nvmlDevice_t, char* name, unsigned int length) {
  if (g_nameResult != ncclSuccess) return g_nameResult;
  snprintf(name, length, "%s", g_deviceName.c_str());
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetMemoryErrorCounter(nvmlDevice_t, nvmlMemoryErrorType_t errorType,
                                                 nvmlEccCounterType_t, nvmlMemoryLocation_t location,
                                                 unsigned long long* count) {
  const int error = ErrorIndex(errorType);
  const int memory = LocationIndex(location);
  if (g_eccResults[error][memory] != ncclSuccess) return g_eccResults[error][memory];
  *count = g_eccCounters[error][memory];
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetNvLinkCapability(nvmlDevice_t, unsigned int link, nvmlNvLinkCapability_t,
                                               unsigned int* capability) {
  if (g_nvLinkCapabilityResults[link] != ncclSuccess) return g_nvLinkCapabilityResults[link];
  *capability = g_nvLinkValid[link];
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetNvLinkState(nvmlDevice_t, unsigned int link, nvmlEnableState_t* state) {
  if (g_nvLinkStateResults[link] != ncclSuccess) return g_nvLinkStateResults[link];
  *state = g_nvLinkStates[link];
  return ncclSuccess;
}

namespace {

struct ReporterState {
  std::vector<std::string> lines;
  ncclResult_t result = ncclSuccess;
};

ncclResult_t CaptureLine(void* target, const char* line) {
  auto* state = static_cast<ReporterState*>(target);
  state->lines.emplace_back(line);
  return state->result;
}

rasDiagnosticsReporter MakeReporter(ReporterState* state) {
  return rasDiagnosticsReporter{CaptureLine, nullptr, state};
}

bool Contains(const std::vector<std::string>& lines, const std::string& text) {
  for (const std::string& line : lines) {
    if (line.find(text) != std::string::npos) return true;
  }
  return false;
}

rasDiagnosticsRankHeader MakeRank(uint64_t commHash, int rank, int nRanks) {
  rasDiagnosticsRankHeader header{};
  header.commId = {commHash, commHash + 1, commHash + 2};
  header.commRank = rank;
  header.commNRanks = nRanks;
  return header;
}

template <typename T>
std::vector<char> BuildRecords(std::initializer_list<std::pair<rasDiagnosticsRankHeader, T>> records) {
  const size_t stride = rasDiagnosticsLocalRecordStride(sizeof(T));
  std::vector<char> data(records.size() * stride, 0);
  size_t offset = 0;
  for (const auto& record : records) {
    memcpy(data.data() + offset, &record.first, sizeof(record.first));
    memcpy(data.data() + offset + sizeof(record.first), &record.second, sizeof(record.second));
    offset += stride;
  }
  return data;
}

rasDiagnosticsGpuModelData GpuModel(int count, const char* model) {
  rasDiagnosticsGpuModelData data{};
  data.nGpus = static_cast<uint8_t>(count);
  snprintf(data.model, sizeof(data.model), "%s", model);
  return data;
}

rasDiagnosticsCudaDriverVersionData DriverVersion(uint32_t version) {
  return rasDiagnosticsCudaDriverVersionData{version};
}

rasDiagnosticsEccData EccData(bool available, uint64_t correctedSram = 0, uint64_t uncorrectedSram = 0,
                              uint64_t correctedDram = 0, uint64_t uncorrectedDram = 0) {
  return rasDiagnosticsEccData{correctedSram, uncorrectedSram, correctedDram, uncorrectedDram,
                               static_cast<uint8_t>(available)};
}

rasDiagnosticsNvLinkData NvLinkData(int links, int inactive) {
  return rasDiagnosticsNvLinkData{static_cast<uint8_t>(links), static_cast<uint8_t>(inactive)};
}

struct OwnedComm {
  std::unique_ptr<ncclComm> comm{new ncclComm{}};
  std::unique_ptr<ncclPeerInfo[]> peers{new ncclPeerInfo[1]{}};

  OwnedComm(uint64_t hash, int rank, int nRanks, int nvmlDev) {
    comm->commHash = hash;
    comm->rank = rank;
    comm->nRanks = nRanks;
    comm->nvmlDev = nvmlDev;
    comm->peerInfoValid = true;
    comm->peerInfo = peers.get();
    peers[0].hostHash = hash + 1;
    peers[0].pidHash = hash + 2;
  }
};

void InstallComms(std::initializer_list<ncclComm*> comms) {
  free(ncclComms);
  ncclComms = static_cast<ncclComm**>(calloc(comms.size(), sizeof(*ncclComms)));
  nNcclComms = static_cast<int>(comms.size());
  int index = 0;
  for (ncclComm* comm : comms) ncclComms[index++] = comm;
}

void ResetState() {
  free(ncclComms);
  ncclComms = nullptr;
  nNcclComms = 0;
  g_eccThreshold = 0;
  g_driverResult = ncclSuccess;
  g_driverVersion = 0;
  g_deviceCountResult = ncclSuccess;
  g_deviceCount = 0;
  g_handleResult = ncclSuccess;
  g_nameResult = ncclSuccess;
  g_deviceName.clear();
  ncclNvmlDeviceCount = 0;
  for (auto& byLocation : g_eccCounters) byLocation.fill(0);
  for (auto& byLocation : g_eccResults) byLocation.fill(ncclSuccess);
  g_nvLinkValid.fill(0);
  g_nvLinkCapabilityResults.fill(ncclSuccess);
  g_nvLinkStates.fill(NVML_FEATURE_ENABLED);
  g_nvLinkStateResults.fill(ncclSuccess);
}

class RasDiagnosticsGpuMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { ResetState(); }
  void TearDown() override { ResetState(); }
};

}  // namespace

TEST_F(RasDiagnosticsGpuMicrotest, SummariesValidateArguments) {
  using SummaryFn = ncclResult_t (*)(const rasDiagnosticsContext*, const rasDiagnosticsReporter*, const char*, int);
  const std::array<SummaryFn, 4> summaries = {rasDiagnosticsGpuModelSummarize,
                                             rasDiagnosticsCudaDriverVersionSummarize,
                                             rasDiagnosticsEccSummarize,
                                             rasDiagnosticsNvLinkSummarize};
  rasDiagnosticsContext ctx{};
  ReporterState state;
  rasDiagnosticsReporter reporter = MakeReporter(&state);
  rasDiagnosticsReporter invalid{};
  const char byte = 0;

  for (SummaryFn summarize : summaries) {
    EXPECT_EQ(ncclInternalError, summarize(&ctx, nullptr, &byte, 1));
    EXPECT_EQ(ncclInternalError, summarize(&ctx, &invalid, &byte, 1));
    EXPECT_EQ(ncclSuccess, summarize(&ctx, &reporter, nullptr, 0));
    EXPECT_EQ(ncclInternalError, summarize(&ctx, &reporter, nullptr, 1));
    EXPECT_EQ(ncclInternalError, summarize(&ctx, &reporter, &byte, -1));
    EXPECT_EQ(ncclInternalError, summarize(&ctx, &reporter, &byte, 1));
  }
}

TEST_F(RasDiagnosticsGpuMicrotest, GpuModelFillReportsKnownAndUnknownInventory) {
  rasDiagnosticsCommSnapshot snapshot{};
  snapshot.nvmlDev = 1;
  ncclNvmlDeviceCount = 2;
  g_deviceCount = 8;
  g_deviceName = "MI300X";
  rasDiagnosticsGpuModelData data{};

  ASSERT_EQ(ncclSuccess, rasDiagnosticsGpuModelFillLocalData(&snapshot, &data));
  EXPECT_EQ(8, data.nGpus);
  EXPECT_STREQ("MI300X", data.model);

  g_deviceCountResult = ncclSystemError;
  g_nameResult = ncclSystemError;
  ASSERT_EQ(ncclSuccess, rasDiagnosticsGpuModelFillLocalData(&snapshot, &data));
  EXPECT_EQ(0, data.nGpus);
  EXPECT_STREQ(RAS_DIAG_GPU_MODEL_UNKNOWN, data.model);

  snapshot.nvmlDev = -1;
  ASSERT_EQ(ncclSuccess, rasDiagnosticsGpuModelFillLocalData(&snapshot, &data));
  EXPECT_STREQ(RAS_DIAG_GPU_MODEL_UNKNOWN, data.model);
}

TEST_F(RasDiagnosticsGpuMicrotest, DriverVersionFillHandlesSuccessFailureAndZero) {
  rasDiagnosticsCommSnapshot snapshot{};
  rasDiagnosticsCudaDriverVersionData data{};
  g_driverVersion = 70002000;
  ASSERT_EQ(ncclSuccess, rasDiagnosticsCudaDriverVersionFillLocalData(&snapshot, &data));
  EXPECT_EQ(70002000u, data.version);

  g_driverVersion = 0;
  ASSERT_EQ(ncclSuccess, rasDiagnosticsCudaDriverVersionFillLocalData(&snapshot, &data));
  EXPECT_EQ(RAS_DIAG_CUDA_DRIVER_VERSION_UNKNOWN, data.version);

  g_driverResult = ncclSystemError;
  ASSERT_EQ(ncclSuccess, rasDiagnosticsCudaDriverVersionFillLocalData(&snapshot, &data));
  EXPECT_EQ(RAS_DIAG_CUDA_DRIVER_VERSION_UNKNOWN, data.version);
}

TEST_F(RasDiagnosticsGpuMicrotest, EccFillRequiresEveryCounter) {
  rasDiagnosticsCommSnapshot snapshot{};
  snapshot.nvmlDev = 0;
  ncclNvmlDeviceCount = 1;
  g_eccCounters = {{{{1, 2}}, {{3, 4}}}};
  rasDiagnosticsEccData data{};

  ASSERT_EQ(ncclSuccess, rasDiagnosticsEccFillLocalData(&snapshot, &data));
  EXPECT_EQ(1u, data.correctedSram);
  EXPECT_EQ(2u, data.correctedDram);
  EXPECT_EQ(3u, data.uncorrectedSram);
  EXPECT_EQ(4u, data.uncorrectedDram);
  EXPECT_EQ(1, data.available);

  g_eccResults[1][1] = ncclSystemError;
  ASSERT_EQ(ncclSuccess, rasDiagnosticsEccFillLocalData(&snapshot, &data));
  EXPECT_EQ(0, data.available);

  g_handleResult = ncclSystemError;
  ASSERT_EQ(ncclSuccess, rasDiagnosticsEccFillLocalData(&snapshot, &data));
  EXPECT_EQ(0, data.available);
}

TEST_F(RasDiagnosticsGpuMicrotest, NvLinkFillCountsValidAndInactiveLinks) {
  rasDiagnosticsCommSnapshot snapshot{};
  snapshot.nvmlDev = 0;
  ncclNvmlDeviceCount = 1;
  g_nvLinkValid[0] = 1;
  g_nvLinkValid[1] = 1;
  g_nvLinkValid[2] = 1;
  g_nvLinkStates[1] = NVML_FEATURE_DISABLED;
  g_nvLinkStateResults[2] = ncclSystemError;
  g_nvLinkCapabilityResults[3] = ncclSystemError;
  rasDiagnosticsNvLinkData data{};

  ASSERT_EQ(ncclSuccess, rasDiagnosticsNvLinkFillLocalData(&snapshot, &data));
  EXPECT_EQ(3, data.nLinks);
  EXPECT_EQ(2, data.nInactive);

  snapshot.nvmlDev = 2;
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNvLinkFillLocalData(&snapshot, &data));
  EXPECT_EQ(0, data.nLinks);
  EXPECT_EQ(0, data.nInactive);
}

TEST_F(RasDiagnosticsGpuMicrotest, CollectLocalBuildsOneRecordForEachCheck) {
  OwnedComm comm(0x100, 0, 1, 0);
  InstallComms({comm.comm.get()});
  ncclNvmlDeviceCount = 1;
  g_deviceCount = 1;
  g_deviceName = "MI300X";
  g_driverVersion = 70002000;
  rasDiagnosticsContext ctx{};

  const std::array<rasDiagnosticsCollectLocalFn, 4> collectors = {rasDiagnosticsGpuModelCollectLocal,
                                                                  rasDiagnosticsCudaDriverVersionCollectLocal,
                                                                  rasDiagnosticsEccCollectLocal,
                                                                  rasDiagnosticsNvLinkCollectLocal};
  for (rasDiagnosticsCollectLocalFn collect : collectors) {
    rasDiagnosticsLocalData data{};
    ASSERT_EQ(ncclSuccess, collect(&ctx, &data));
    EXPECT_EQ(1, data.nRecords);
    EXPECT_GT(data.recordsBytes, static_cast<int>(sizeof(rasDiagnosticsRankHeader)));
    free(data.records);
  }
}

TEST_F(RasDiagnosticsGpuMicrotest, GpuModelSummaryCoversAvailabilityAndMismatchCases) {
  rasDiagnosticsContext ctx{};
  ReporterState state;
  rasDiagnosticsReporter reporter = MakeReporter(&state);

  auto records = BuildRecords<rasDiagnosticsGpuModelData>(
    {{MakeRank(1, 0, 1), GpuModel(8, "MI300X")},
     {MakeRank(2, 0, 1), GpuModel(8, RAS_DIAG_GPU_MODEL_UNKNOWN)},
     {MakeRank(3, 0, 1), GpuModel(0, "MI250")},
     {MakeRank(4, 0, 1), GpuModel(0, RAS_DIAG_GPU_MODEL_UNKNOWN)}});
  ASSERT_EQ(ncclSuccess,
            rasDiagnosticsGpuModelSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(Contains(state.lines, "8x MI300X"));
  EXPECT_TRUE(Contains(state.lines, "GPU model unavailable"));
  EXPECT_TRUE(Contains(state.lines, "GPU count unavailable"));
  EXPECT_TRUE(Contains(state.lines, "unavailable via NVML"));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsGpuModelData>(
    {{MakeRank(5, 0, 2), GpuModel(8, "MI300X")}, {MakeRank(5, 1, 2), GpuModel(4, "MI250")}});
  ASSERT_EQ(ncclSuccess,
            rasDiagnosticsGpuModelSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(Contains(state.lines, "count mismatch"));
  EXPECT_TRUE(Contains(state.lines, "model mismatch"));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsGpuModelData>({{MakeRank(6, 0, 2), GpuModel(8, "MI300X")}});
  ASSERT_EQ(ncclSuccess,
            rasDiagnosticsGpuModelSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(Contains(state.lines, "diagnostics incomplete"));
}

TEST_F(RasDiagnosticsGpuMicrotest, DriverVersionSummaryCoversKnownUnknownMismatchAndIncomplete) {
  rasDiagnosticsContext ctx{};
  ReporterState state;
  rasDiagnosticsReporter reporter = MakeReporter(&state);
  auto records = BuildRecords<rasDiagnosticsCudaDriverVersionData>(
    {{MakeRank(1, 0, 1), DriverVersion(70002000)},
     {MakeRank(2, 0, 1), DriverVersion(RAS_DIAG_CUDA_DRIVER_VERSION_UNKNOWN)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsCudaDriverVersionSummarize(&ctx, &reporter, records.data(), records.size()));
  EXPECT_TRUE(Contains(state.lines, "70002000 consistent"));
  EXPECT_TRUE(Contains(state.lines, "unavailable across"));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsCudaDriverVersionData>(
    {{MakeRank(3, 0, 2), DriverVersion(1)}, {MakeRank(3, 1, 2), DriverVersion(2)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsCudaDriverVersionSummarize(&ctx, &reporter, records.data(), records.size()));
  EXPECT_TRUE(Contains(state.lines, "mismatch"));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsCudaDriverVersionData>({{MakeRank(4, 0, 2), DriverVersion(1)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsCudaDriverVersionSummarize(&ctx, &reporter, records.data(), records.size()));
  EXPECT_TRUE(Contains(state.lines, "diagnostics incomplete"));
}

TEST_F(RasDiagnosticsGpuMicrotest, EccSummaryCoversAvailabilityHealthyAndErrorCases) {
  rasDiagnosticsContext ctx{};
  ReporterState state;
  rasDiagnosticsReporter reporter = MakeReporter(&state);

  auto records = BuildRecords<rasDiagnosticsEccData>({{MakeRank(1, 0, 1), EccData(false)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsEccSummarize(&ctx, &reporter, records.data(), records.size()));
  EXPECT_TRUE(Contains(state.lines, "unavailable via NVML"));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsEccData>({{MakeRank(2, 0, 1), EccData(true)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsEccSummarize(&ctx, &reporter, records.data(), records.size()));
  EXPECT_TRUE(Contains(state.lines, "no uncorrected"));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsEccData>(
    {{MakeRank(3, 0, 2), EccData(true)}, {MakeRank(3, 1, 2), EccData(false)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsEccSummarize(&ctx, &reporter, records.data(), records.size()));
  EXPECT_TRUE(Contains(state.lines, "across 1 of 2 ranks"));

  state.lines.clear();
  g_eccThreshold = 5;
  records = BuildRecords<rasDiagnosticsEccData>(
    {{MakeRank(4, 0, 2), EccData(true, 3, 1, 4, 2)}, {MakeRank(4, 1, 2), EccData(true)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsEccSummarize(&ctx, &reporter, records.data(), records.size()));
  EXPECT_TRUE(Contains(state.lines, "uncorrected volatile errors"));
  EXPECT_TRUE(Contains(state.lines, "corrected volatile errors"));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsEccData>({{MakeRank(5, 0, 2), EccData(true)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsEccSummarize(&ctx, &reporter, records.data(), records.size()));
  EXPECT_TRUE(Contains(state.lines, "diagnostics incomplete"));
}

TEST_F(RasDiagnosticsGpuMicrotest, NvLinkSummaryCoversNoLinksHealthyMismatchInactiveAndIncomplete) {
  rasDiagnosticsContext ctx{};
  ReporterState state;
  rasDiagnosticsReporter reporter = MakeReporter(&state);

  auto records = BuildRecords<rasDiagnosticsNvLinkData>({{MakeRank(1, 0, 1), NvLinkData(0, 0)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNvLinkSummarize(&ctx, &reporter, records.data(), records.size()));
  EXPECT_TRUE(state.lines.empty());

  records = BuildRecords<rasDiagnosticsNvLinkData>({{MakeRank(2, 0, 1), NvLinkData(8, 0)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNvLinkSummarize(&ctx, &reporter, records.data(), records.size()));
  EXPECT_TRUE(Contains(state.lines, "all active"));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsNvLinkData>(
    {{MakeRank(3, 0, 2), NvLinkData(8, 0)}, {MakeRank(3, 1, 2), NvLinkData(6, 1)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNvLinkSummarize(&ctx, &reporter, records.data(), records.size()));
  EXPECT_TRUE(Contains(state.lines, "link-count mismatch"));
  EXPECT_TRUE(Contains(state.lines, "inactive link"));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsNvLinkData>({{MakeRank(4, 0, 2), NvLinkData(8, 0)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNvLinkSummarize(&ctx, &reporter, records.data(), records.size()));
  EXPECT_TRUE(Contains(state.lines, "diagnostics incomplete"));
}

TEST_F(RasDiagnosticsGpuMicrotest, SummaryPropagatesReporterFailure) {
  rasDiagnosticsContext ctx{};
  ReporterState state;
  state.result = ncclRemoteError;
  rasDiagnosticsReporter reporter = MakeReporter(&state);
  auto records = BuildRecords<rasDiagnosticsGpuModelData>({{MakeRank(1, 0, 1), GpuModel(8, "MI300X")}});

  EXPECT_EQ(ncclRemoteError,
            rasDiagnosticsGpuModelSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
}
