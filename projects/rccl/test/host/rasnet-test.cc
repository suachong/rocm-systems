/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ras/rasnet.cc.

#include <arpa/inet.h>
#include <poll.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#define rasNextLink RasNetTestNextLink
#define rasPrevLink RasNetTestPrevLink
#define rasConnsHead RasNetTestConnsHead
#define rasConnsTail RasNetTestConnsTail
#define rasSocketsHead RasNetTestSocketsHead
#define rasSocketsTail RasNetTestSocketsTail
#define getNewConnEntry RasNetTestGetNewConnEntry
#define rasConnCreate RasNetTestConnCreate
#define rasConnFind RasNetTestConnFind
#define rasConnsHandleTimeouts RasNetTestConnsHandleTimeouts
#define rasConnDisconnect RasNetTestConnDisconnect
#define rasNetAcceptNewSocket RasNetTestAcceptNewSocket
#define rasSocksHandleTimeouts RasNetTestSocksHandleTimeouts
#define rasSocketTerminate RasNetTestSocketTerminate
#define rasSockEventLoop RasNetTestSockEventLoop
#define rasNetHandleTimeouts RasNetTestHandleTimeouts
#define rasMsgHandleKeepAlive RasNetTestMsgHandleKeepAlive
#define rasLinkAddFallback RasNetTestLinkAddFallback
#define rasLinkConnUpdate RasNetTestLinkConnUpdate
#define rasNetTerminate RasNetTestTerminate

#define rasPfds RasNetTestPfds
#define rasNetListeningSocket RasNetTestListeningSocket
#define rasLine RasNetTestLine
#define rasPeers RasNetTestPeers
#define nRasPeers RasNetTestNPeers
#define rasPeersHash RasNetTestPeersHash
#define rasDeadPeersHash RasNetTestDeadPeersHash
#define rasMsgAlloc RasNetTestMsgAlloc
#define rasConnEnqueueMsg RasNetTestConnEnqueueMsg
#define rasConnSendMsg RasNetTestConnSendMsg
#define rasMsgRecv RasNetTestMsgRecv
#define rasMsgHandle RasNetTestMsgHandle
#define rasGetNewPollEntry RasNetTestGetNewPollEntry
#define rasClientsNotifyEvent RasNetTestClientsNotifyEvent
#define rasCollReqInit RasNetTestCollReqInit
#define rasNetSendCollReq RasNetTestSendCollReq
#define rasCollsPurgeConn RasNetTestCollsPurgeConn
#define rasConnSendPeersUpdate RasNetTestConnSendPeersUpdate
#define rasPeerFind RasNetTestPeerFind
#define rasLinkCalculatePeer RasNetTestLinkCalculatePeer
#define rasTimeoutFactorNs RasNetTestTimeoutFactorNs

#define ncclSocketDefaultMagic RasNetTestSocketDefaultMagic
#define ncclSocketInit RasNetTestSocketInit
#define ncclSocketConnect RasNetTestSocketConnect
#define ncclSocketReady RasNetTestSocketReady
#define ncclSocketAccept RasNetTestSocketAccept
#define ncclSocketShutdown RasNetTestSocketShutdown
#define ncclSocketClose RasNetTestSocketClose
#define ncclSocketToString RasNetTestSocketToString

#include "alloc.h"
#include "ras/ras_internal.h"

namespace {

int g_callocCalls;
int g_failCallocAt;

template <typename T>
ncclResult_t RasNetTestCalloc(T** ptr, size_t count) {
  if (g_callocCalls++ == g_failCallocAt) return ncclSystemError;
  *ptr = static_cast<T*>(calloc(count, sizeof(T)));
  return *ptr ? ncclSuccess : ncclSystemError;
}

int64_t g_clockNano;
timespec g_realtime;

}  // namespace

uint64_t RasNetTestClockNano();
void RasNetTestClockRealtime(struct timespec* time);

#undef ncclCalloc
#define ncclCalloc(...) RasNetTestCalloc(__VA_ARGS__)
#define clockNano RasNetTestClockNano
#define clockRealtime RasNetTestClockRealtime

#include RASNET_CC_PATH

#undef clockRealtime
#undef clockNano
#undef ncclCalloc

namespace {

struct EventRecord {
  rasEventGroup group;
  std::string type;
  std::string details;
  bool hasAddr;
  ncclSocketAddress addr;
};

struct RecvStep {
  ncclResult_t result;
  int closed;
  rasMsg* msg;
};

std::array<pollfd, 32> g_pollFds;
int g_nextPollEntry;
ncclResult_t g_pollEntryResult;

ncclResult_t g_socketInitResult;
ncclResult_t g_socketConnectResult;
ncclResult_t g_socketReadyResult;
ncclResult_t g_socketAcceptResult;
ncclResult_t g_socketShutdownResult;
ncclResult_t g_socketCloseResult;
int g_socketReadyValue;
ncclSocketState g_socketReadyState;
ncclSocketDescriptor g_nextSocketFd;
ncclSocketDescriptor g_acceptSocketFd;
int g_socketInitCalls;
int g_socketConnectCalls;
int g_socketReadyCalls;
int g_socketAcceptCalls;
int g_socketShutdownCalls;
int g_socketCloseCalls;
int g_lastShutdownHow;

ncclResult_t g_msgAllocResult;
std::vector<rasConnection*> g_enqueuedConnections;
std::vector<bool> g_enqueuedFront;
ncclResult_t g_sendMsgResult;
int g_sendMsgClosed;
bool g_sendMsgAllSent;
int g_sendMsgCalls;
std::deque<RecvStep> g_recvSteps;
int g_recvCalls;
ncclResult_t g_handleMsgResult;
int g_handleMsgCalls;
rasMsgType g_lastHandledType;
bool g_clearPollInOnHandle;

std::vector<EventRecord> g_events;
std::vector<rasConnection*> g_purgedConnections;
int g_collReqInitCalls;
int g_sendCollReqCalls;
ncclResult_t g_sendCollReqResult;
rasCollRequest g_lastCollReq;
int g_sendPeersUpdateCalls;
ncclResult_t g_sendPeersUpdateResult;
rasConnection* g_lastPeersUpdateConn;

int g_peerFindResult;
std::deque<int> g_calculatedPeers;

ncclSocketAddress MakeAddr(uint16_t port) {
  ncclSocketAddress addr{};
  addr.sin.sin_family = AF_INET;
  addr.sin.sin_port = htons(port);
  addr.sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  return addr;
}

rasMsgMeta* MetaFromMsg(rasMsg* msg) {
  return reinterpret_cast<rasMsgMeta*>(reinterpret_cast<char*>(msg) - offsetof(rasMsgMeta, msg));
}

rasMsg* MakeRecvMsg(rasMsgType type) {
  auto* msg = static_cast<rasMsg*>(calloc(1, rasMsgLength(type)));
  msg->type = type;
  return msg;
}

rasMsgMeta* MakeQueuedMsg(rasConnection* conn, rasMsgType type, int64_t enqueueTime, int offset = 0) {
  rasMsg* msg = nullptr;
  EXPECT_EQ(ncclSuccess, rasMsgAlloc(&msg, rasMsgLength(type)));
  msg->type = type;
  rasMsgMeta* meta = MetaFromMsg(msg);
  meta->enqueueTime = enqueueTime;
  meta->offset = offset;
  meta->length = static_cast<int>(rasMsgLength(type));
  ncclIntruQueueEnqueue(&conn->sendQ, meta);
  return meta;
}

rasConnection* MakeConn(uint16_t port = 1000, bool link = true) {
  auto* conn = static_cast<rasConnection*>(calloc(1, sizeof(rasConnection)));
  conn->addr = MakeAddr(port);
  conn->travelTimeMin = INT64_MAX;
  conn->travelTimeMax = INT64_MIN;
  ncclIntruQueueConstruct(&conn->sendQ);
  if (link) {
    conn->prev = rasConnsTail;
    if (rasConnsTail) rasConnsTail->next = conn;
    else rasConnsHead = conn;
    rasConnsTail = conn;
  }
  return conn;
}

rasSocket* MakeSock(rasConnection* conn = nullptr, rasSocketStatus status = RAS_SOCK_READY, int pfd = 0,
                    bool link = true) {
  auto* sock = static_cast<rasSocket*>(calloc(1, sizeof(rasSocket)));
  sock->conn = conn;
  sock->status = status;
  sock->pfd = pfd;
  sock->sock.socketDescriptor = 100 + pfd;
  sock->sock.state = ncclSocketStateReady;
  if (conn) conn->sock = sock;
  if (link) {
    sock->prev = rasSocketsTail;
    if (rasSocketsTail) rasSocketsTail->next = sock;
    else rasSocketsHead = sock;
    rasSocketsTail = sock;
  }
  rasPfds[pfd].fd = sock->sock.socketDescriptor;
  rasPfds[pfd].events = POLLIN | POLLOUT;
  return sock;
}

rasLinkConn* AddLinkConn(rasLink* link, rasConnection* conn, int peerIdx, bool external = false) {
  auto* entry = static_cast<rasLinkConn*>(calloc(1, sizeof(rasLinkConn)));
  entry->conn = conn;
  entry->peerIdx = peerIdx;
  entry->external = external;
  if (link->conns == nullptr) link->conns = entry;
  else {
    rasLinkConn* tail = link->conns;
    while (tail->next) tail = tail->next;
    tail->next = entry;
  }
  return entry;
}

std::vector<int> LinkPeerIndices(const rasLink& link) {
  std::vector<int> indices;
  for (rasLinkConn* entry = link.conns; entry; entry = entry->next) indices.push_back(entry->peerIdx);
  return indices;
}

void SetPeers(std::initializer_list<uint16_t> ports) {
  free(rasPeers);
  nRasPeers = static_cast<int>(ports.size());
  rasPeers = static_cast<rasPeerInfo*>(calloc(std::max(1, nRasPeers), sizeof(*rasPeers)));
  int index = 0;
  for (uint16_t port : ports) rasPeers[index++].addr = MakeAddr(port);
}

void FreeLink(rasLink* link) {
  while (link->conns) {
    rasLinkConn* next = link->conns->next;
    free(link->conns);
    link->conns = next;
  }
  link->lastUpdatePeersTime = 0;
}

void FreeState() {
  FreeLink(&rasNextLink);
  FreeLink(&rasPrevLink);
  while (rasSocketsHead) {
    rasSocket* next = rasSocketsHead->next;
    free(rasSocketsHead->recvMsg);
    free(rasSocketsHead);
    rasSocketsHead = next;
  }
  rasSocketsTail = nullptr;
  while (rasConnsHead) {
    rasConnection* next = rasConnsHead->next;
    while (rasMsgMeta* meta = ncclIntruQueueTryDequeue(&rasConnsHead->sendQ)) free(meta);
    free(rasConnsHead);
    rasConnsHead = next;
  }
  rasConnsTail = nullptr;
  for (RecvStep& step : g_recvSteps) free(step.msg);
  g_recvSteps.clear();
  free(rasPeers);
  rasPeers = nullptr;
  nRasPeers = 0;
}

void ResetState() {
  FreeState();
  rasNextLink = {1};
  rasPrevLink = {-1};
  std::memset(&rasNetListeningSocket, 0, sizeof(rasNetListeningSocket));
  std::memset(rasLine, 0, sizeof(rasLine));
  g_pollFds.fill(pollfd{NCCL_INVALID_SOCKET, 0, 0});
  rasPfds = g_pollFds.data();
  g_nextPollEntry = 0;
  g_pollEntryResult = ncclSuccess;
  g_callocCalls = 0;
  g_failCallocAt = -1;
  g_clockNano = 100 * CLOCK_UNITS_PER_SEC;
  g_realtime = {123, 456};
  g_socketInitResult = ncclSuccess;
  g_socketConnectResult = ncclSuccess;
  g_socketReadyResult = ncclSuccess;
  g_socketAcceptResult = ncclSuccess;
  g_socketShutdownResult = ncclSuccess;
  g_socketCloseResult = ncclSuccess;
  g_socketReadyValue = 1;
  g_socketReadyState = ncclSocketStateReady;
  g_nextSocketFd = 40;
  g_acceptSocketFd = 41;
  g_socketInitCalls = 0;
  g_socketConnectCalls = 0;
  g_socketReadyCalls = 0;
  g_socketAcceptCalls = 0;
  g_socketShutdownCalls = 0;
  g_socketCloseCalls = 0;
  g_lastShutdownHow = -1;
  g_msgAllocResult = ncclSuccess;
  g_enqueuedConnections.clear();
  g_enqueuedFront.clear();
  g_sendMsgResult = ncclSuccess;
  g_sendMsgClosed = 0;
  g_sendMsgAllSent = false;
  g_sendMsgCalls = 0;
  g_recvCalls = 0;
  g_handleMsgResult = ncclSuccess;
  g_handleMsgCalls = 0;
  g_lastHandledType = RAS_MSG_CONNINIT;
  g_clearPollInOnHandle = false;
  g_events.clear();
  g_purgedConnections.clear();
  g_collReqInitCalls = 0;
  g_sendCollReqCalls = 0;
  g_sendCollReqResult = ncclSuccess;
  g_lastCollReq = {};
  g_sendPeersUpdateCalls = 0;
  g_sendPeersUpdateResult = ncclSuccess;
  g_lastPeersUpdateConn = nullptr;
  g_peerFindResult = -1;
  g_calculatedPeers.clear();
  rasPeersHash = 0x1111;
  rasDeadPeersHash = 0x2222;
}

class RasNetMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { ResetState(); }
  void TearDown() override { FreeState(); }
};

}  // namespace

uint64_t RasNetTestClockNano() { return static_cast<uint64_t>(g_clockNano); }
void RasNetTestClockRealtime(struct timespec* time) { *time = g_realtime; }
int64_t rasTimeoutFactorNs(int64_t baseSeconds) { return baseSeconds * CLOCK_UNITS_PER_SEC; }

struct pollfd* rasPfds;
struct ncclSocket rasNetListeningSocket;
char rasLine[SOCKET_NAME_MAXLEN + 1];
struct rasPeerInfo* rasPeers;
int nRasPeers;
uint64_t rasPeersHash;
uint64_t rasDeadPeersHash;

uint64_t ncclSocketDefaultMagic() { return 0x12345678; }

ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic,
                            enum ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag, int customRetry) {
  ++g_socketInitCalls;
  if (g_socketInitResult != ncclSuccess) return g_socketInitResult;
  std::memset(sock, 0, sizeof(*sock));
  if (addr) sock->addr = *addr;
  sock->magic = magic;
  sock->type = type;
  sock->abortFlag = abortFlag;
  sock->asyncFlag = asyncFlag;
  sock->customRetry = customRetry;
  sock->socketDescriptor = NCCL_INVALID_SOCKET;
  sock->state = ncclSocketStateInitialized;
  return ncclSuccess;
}

ncclResult_t ncclSocketConnect(struct ncclSocket* sock) {
  ++g_socketConnectCalls;
  if (g_socketConnectResult != ncclSuccess) return g_socketConnectResult;
  sock->socketDescriptor = g_nextSocketFd;
  sock->state = g_socketReadyState;
  return ncclSuccess;
}

ncclResult_t ncclSocketReady(struct ncclSocket* sock, int* ready) {
  ++g_socketReadyCalls;
  if (g_socketReadyResult != ncclSuccess) return g_socketReadyResult;
  *ready = g_socketReadyValue;
  sock->state = g_socketReadyState;
  return ncclSuccess;
}

ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket*, bool) {
  ++g_socketAcceptCalls;
  if (g_socketAcceptResult != ncclSuccess) return g_socketAcceptResult;
  sock->socketDescriptor = g_acceptSocketFd;
  sock->state = ncclSocketStateAccepted;
  sock->addr = MakeAddr(9000);
  return ncclSuccess;
}

ncclResult_t ncclSocketShutdown(struct ncclSocket*, int how) {
  ++g_socketShutdownCalls;
  g_lastShutdownHow = how;
  return g_socketShutdownResult;
}

ncclResult_t ncclSocketClose(struct ncclSocket* sock, bool) {
  ++g_socketCloseCalls;
  sock->socketDescriptor = NCCL_INVALID_SOCKET;
  sock->state = ncclSocketStateClosed;
  return g_socketCloseResult;
}

const char* ncclSocketToString(const union ncclSocketAddress* addr, char* buf, const int) {
  std::snprintf(buf, SOCKET_NAME_MAXLEN + 1, "port-%u", addr ? ntohs(addr->sin.sin_port) : 0);
  return buf;
}

ncclResult_t rasGetNewPollEntry(int* index) {
  if (g_pollEntryResult != ncclSuccess) return g_pollEntryResult;
  *index = g_nextPollEntry++;
  rasPfds[*index] = {NCCL_INVALID_SOCKET, 0, 0};
  return ncclSuccess;
}

ncclResult_t rasMsgAlloc(struct rasMsg** msg, size_t msgLen) {
  if (g_msgAllocResult != ncclSuccess) return g_msgAllocResult;
  auto* meta = static_cast<rasMsgMeta*>(calloc(1, offsetof(rasMsgMeta, msg) + msgLen));
  if (meta == nullptr) return ncclSystemError;
  *msg = &meta->msg;
  return ncclSuccess;
}

void rasConnEnqueueMsg(struct rasConnection* conn, struct rasMsg* msg, size_t msgLen, bool front) {
  rasMsgMeta* meta = MetaFromMsg(msg);
  meta->enqueueTime = g_clockNano;
  meta->length = static_cast<int>(msgLen);
  if (front) ncclIntruQueueEnqueueFront(&conn->sendQ, meta);
  else ncclIntruQueueEnqueue(&conn->sendQ, meta);
  g_enqueuedConnections.push_back(conn);
  g_enqueuedFront.push_back(front);
}

ncclResult_t rasConnSendMsg(struct rasConnection*, int* closed, bool* allSent) {
  ++g_sendMsgCalls;
  *closed = g_sendMsgClosed;
  *allSent = g_sendMsgAllSent;
  return g_sendMsgResult;
}

ncclResult_t rasMsgRecv(struct rasSocket*, struct rasMsg** msg, int* closed) {
  ++g_recvCalls;
  if (g_recvSteps.empty()) {
    *msg = nullptr;
    *closed = 0;
    return ncclSuccess;
  }
  RecvStep step = g_recvSteps.front();
  g_recvSteps.pop_front();
  *msg = step.msg;
  *closed = step.closed;
  return step.result;
}

ncclResult_t rasMsgHandle(struct rasMsg* msg, struct rasSocket* sock) {
  ++g_handleMsgCalls;
  g_lastHandledType = msg->type;
  if (g_clearPollInOnHandle) rasPfds[sock->pfd].revents &= ~POLLIN;
  return g_handleMsgResult;
}

void rasClientsNotifyEvent(rasEventGroup group, const struct rasEventNotification* event) {
  EventRecord record{};
  record.group = group;
  record.type = event && event->eventType ? event->eventType : "";
  record.details = event && event->details ? event->details : "";
  record.hasAddr = event && event->peerAddr;
  if (record.hasAddr) record.addr = *event->peerAddr;
  g_events.push_back(record);
}

void rasCollReqInit(struct rasCollRequest* req) {
  ++g_collReqInitCalls;
  req->rootId = 77;
}

ncclResult_t rasNetSendCollReq(const struct rasCollRequest* req, bool*, struct rasCollective**, struct rasConnection*) {
  ++g_sendCollReqCalls;
  g_lastCollReq = *req;
  return g_sendCollReqResult;
}

void rasCollsPurgeConn(struct rasConnection* conn) { g_purgedConnections.push_back(conn); }

ncclResult_t rasConnSendPeersUpdate(struct rasConnection* conn, const struct rasPeerInfo*, int) {
  ++g_sendPeersUpdateCalls;
  g_lastPeersUpdateConn = conn;
  return g_sendPeersUpdateResult;
}

int rasPeerFind(const union ncclSocketAddress*) { return g_peerFindResult; }

int rasLinkCalculatePeer(const struct rasLink*, int, bool) {
  if (g_calculatedPeers.empty()) return -1;
  int peer = g_calculatedPeers.front();
  g_calculatedPeers.pop_front();
  return peer;
}

TEST_F(RasNetMicrotest, ConnectionEntriesInitializeAppendFindAndUnlink) {
  rasConnection* first = nullptr;
  rasConnection* second = nullptr;
  ASSERT_EQ(ncclSuccess, getNewConnEntry(&first));
  ASSERT_EQ(ncclSuccess, getNewConnEntry(&second));
  first->addr = MakeAddr(1001);
  second->addr = MakeAddr(1002);
  EXPECT_EQ(INT64_MAX, first->travelTimeMin);
  EXPECT_EQ(INT64_MIN, first->travelTimeMax);
  EXPECT_EQ(first, rasConnsHead);
  EXPECT_EQ(second, rasConnsTail);
  EXPECT_EQ(second, first->next);
  EXPECT_EQ(first, second->prev);
  EXPECT_EQ(first, rasConnFind(&first->addr));
  ncclSocketAddress absent = MakeAddr(1999);
  EXPECT_EQ(nullptr, rasConnFind(&absent));
  freeConnEntry(first);
  EXPECT_EQ(second, rasConnsHead);
  EXPECT_EQ(nullptr, second->prev);
  freeConnEntry(nullptr);
  freeConnEntry(second);
  EXPECT_EQ(nullptr, rasConnsHead);
  EXPECT_EQ(nullptr, rasConnsTail);
}

TEST_F(RasNetMicrotest, ConnectionAllocationFailurePropagates) {
  g_failCallocAt = 0;
  rasConnection* conn = nullptr;
  EXPECT_EQ(ncclSystemError, getNewConnEntry(&conn));
  EXPECT_EQ(nullptr, conn);
}

TEST_F(RasNetMicrotest, ConnectionCreateReusesLiveEntry) {
  rasConnection* conn = MakeConn(1001);
  rasSocket* sock = MakeSock(conn);
  rasConnection* result = nullptr;
  EXPECT_EQ(ncclSuccess, rasConnCreate(&conn->addr, &result));
  EXPECT_EQ(conn, result);
  EXPECT_EQ(sock, conn->sock);
  EXPECT_EQ(0, g_socketInitCalls);
}

TEST_F(RasNetMicrotest, ConnectionCreateOpensAndTracksNewEntry) {
  ncclSocketAddress addr = MakeAddr(1002);
  rasConnection* conn = nullptr;
  ASSERT_EQ(ncclSuccess, rasConnCreate(&addr, &conn));
  ASSERT_NE(nullptr, conn);
  ASSERT_NE(nullptr, conn->sock);
  EXPECT_EQ(0, std::memcmp(&addr, &conn->addr, sizeof(addr)));
  EXPECT_EQ(g_clockNano, conn->startRetryTime);
  EXPECT_EQ(g_clockNano, conn->lastRetryTime);
  EXPECT_EQ(RAS_SOCK_CONNECTING, conn->sock->status);
  EXPECT_EQ(POLLIN | POLLOUT, rasPfds[conn->sock->pfd].events);
  EXPECT_EQ(g_nextSocketFd, rasPfds[conn->sock->pfd].fd);
}

TEST_F(RasNetMicrotest, ConnectionOpenIgnoresFailuresAndCleansPartialSocket) {
  rasConnection* conn = MakeConn();
  g_socketConnectResult = ncclSystemError;
  rasConnOpen(conn);
  EXPECT_EQ(nullptr, conn->sock);
  EXPECT_EQ(1, g_socketCloseCalls);
  EXPECT_EQ(nullptr, rasSocketsHead);
  EXPECT_EQ(g_clockNano, conn->lastRetryTime);

  g_socketConnectResult = ncclSuccess;
  g_failCallocAt = g_callocCalls;
  rasConnOpen(conn);
  EXPECT_EQ(nullptr, rasSocketsHead);
  EXPECT_EQ(1, g_socketCloseCalls);
}

TEST_F(RasNetMicrotest, ConnectionOpenHandlesReadyAndPollFailures) {
  rasConnection* conn = MakeConn();
  g_socketReadyResult = ncclSystemError;
  rasConnOpen(conn);
  EXPECT_EQ(1, g_socketCloseCalls);
  EXPECT_EQ(nullptr, rasSocketsHead);

  g_socketReadyResult = ncclSuccess;
  g_pollEntryResult = ncclSystemError;
  rasConnOpen(conn);
  EXPECT_EQ(2, g_socketCloseCalls);
  EXPECT_EQ(nullptr, rasSocketsHead);
}

TEST_F(RasNetMicrotest, ConnectionOpenIgnoresConnectingDescriptor) {
  rasConnection* conn = MakeConn();
  g_socketReadyState = ncclSocketStateConnecting;
  rasConnOpen(conn);
  ASSERT_NE(nullptr, conn->sock);
  EXPECT_EQ(POLL_FD_IGNORE, rasPfds[conn->sock->pfd].fd);
}

TEST_F(RasNetMicrotest, ConnectionPrepareBuildsFrontConnInit) {
  rasConnection* conn = MakeConn();
  rasNetListeningSocket.addr = MakeAddr(4321);
  ASSERT_EQ(ncclSuccess, rasConnPrepare(conn));
  ASSERT_EQ(1u, g_enqueuedConnections.size());
  EXPECT_EQ(conn, g_enqueuedConnections[0]);
  EXPECT_TRUE(g_enqueuedFront[0]);
  rasMsgMeta* meta = ncclIntruQueueHead(&conn->sendQ);
  EXPECT_EQ(RAS_MSG_CONNINIT, meta->msg.type);
  EXPECT_EQ(NCCL_VERSION_CODE, meta->msg.connInit.ncclVersion);
  EXPECT_EQ(rasPeersHash, meta->msg.connInit.peersHash);
  EXPECT_EQ(rasDeadPeersHash, meta->msg.connInit.deadPeersHash);
  EXPECT_EQ(htons(4321), meta->msg.connInit.listeningAddr.sin.sin_port);
}

TEST_F(RasNetMicrotest, ConnectionPreparePropagatesAllocationFailure) {
  rasConnection* conn = MakeConn();
  g_msgAllocResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, rasConnPrepare(conn));
  EXPECT_TRUE(ncclIntruQueueEmpty(&conn->sendQ));
}

TEST_F(RasNetMicrotest, AcceptNewSocketInitializesPollState) {
  ASSERT_EQ(ncclSuccess, rasNetAcceptNewSocket());
  ASSERT_NE(nullptr, rasSocketsHead);
  EXPECT_EQ(RAS_SOCK_CONNECTING, rasSocketsHead->status);
  EXPECT_EQ(g_acceptSocketFd, rasPfds[rasSocketsHead->pfd].fd);
  EXPECT_EQ(POLLIN, rasPfds[rasSocketsHead->pfd].events);
}

TEST_F(RasNetMicrotest, AcceptNewSocketCleansEveryFailureStage) {
  g_socketInitResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, rasNetAcceptNewSocket());
  EXPECT_EQ(0, g_socketCloseCalls);

  g_socketInitResult = ncclSuccess;
  g_socketAcceptResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, rasNetAcceptNewSocket());
  EXPECT_EQ(1, g_socketCloseCalls);

  g_socketAcceptResult = ncclSuccess;
  g_socketReadyResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, rasNetAcceptNewSocket());
  EXPECT_EQ(2, g_socketCloseCalls);

  g_socketReadyResult = ncclSuccess;
  g_acceptSocketFd = NCCL_INVALID_SOCKET;
  EXPECT_EQ(ncclSuccess, rasNetAcceptNewSocket());
  EXPECT_EQ(3, g_socketCloseCalls);

  g_acceptSocketFd = 44;
  g_pollEntryResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, rasNetAcceptNewSocket());
  EXPECT_EQ(4, g_socketCloseCalls);
  EXPECT_EQ(nullptr, rasSocketsHead);
}

TEST_F(RasNetMicrotest, SocketEntriesInitializeAppendAndUnlink) {
  rasSocket* first = nullptr;
  rasSocket* second = nullptr;
  ASSERT_EQ(ncclSuccess, getNewSockEntry(&first));
  ++g_clockNano;
  ASSERT_EQ(ncclSuccess, getNewSockEntry(&second));
  EXPECT_EQ(-1, first->pfd);
  EXPECT_LT(first->createTime, second->createTime);
  EXPECT_EQ(first, rasSocketsHead);
  EXPECT_EQ(second, rasSocketsTail);
  freeSockEntry(first);
  EXPECT_EQ(second, rasSocketsHead);
  freeSockEntry(nullptr);
  freeSockEntry(second);
  EXPECT_EQ(nullptr, rasSocketsHead);
  EXPECT_EQ(nullptr, rasSocketsTail);
}

TEST_F(RasNetMicrotest, SocketAllocationFailurePropagates) {
  g_failCallocAt = 0;
  rasSocket* sock = nullptr;
  EXPECT_EQ(ncclSystemError, getNewSockEntry(&sock));
}

TEST_F(RasNetMicrotest, SocketTimeoutsScheduleUnexpiredStates) {
  rasSocket* connecting = MakeSock(nullptr, RAS_SOCK_CONNECTING, 0);
  connecting->createTime = 90 * CLOCK_UNITS_PER_SEC;
  rasSocket* terminating = MakeSock(nullptr, RAS_SOCK_TERMINATING, 1);
  terminating->lastSendTime = 91 * CLOCK_UNITS_PER_SEC;
  terminating->lastRecvTime = 92 * CLOCK_UNITS_PER_SEC;
  rasSocket* ready = MakeSock(nullptr, RAS_SOCK_READY, 2);
  ready->lastSendTime = 80 * CLOCK_UNITS_PER_SEC;
  ready->lastRecvTime = 81 * CLOCK_UNITS_PER_SEC;
  int64_t nextWakeup = INT64_MAX;
  rasSocksHandleTimeouts(100 * CLOCK_UNITS_PER_SEC, &nextWakeup);
  EXPECT_EQ(connecting->createTime + RAS_STUCK_TIMEOUT, nextWakeup);
  EXPECT_EQ(0, g_socketCloseCalls);
}

TEST_F(RasNetMicrotest, SocketTimeoutsTerminateExpiredStates) {
  rasSocket* incoming = MakeSock(nullptr, RAS_SOCK_CONNECTING, 0);
  incoming->createTime = 1;
  rasConnection* conn = MakeConn();
  rasSocket* outgoing = MakeSock(conn, RAS_SOCK_HANDSHAKE, 1);
  outgoing->createTime = 1;
  conn->startRetryTime = 2;
  rasSocket* terminating = MakeSock(nullptr, RAS_SOCK_TERMINATING, 2);
  terminating->lastSendTime = terminating->lastRecvTime = 1;
  rasSocket* idle = MakeSock(nullptr, RAS_SOCK_READY, 3);
  idle->lastSendTime = idle->lastRecvTime = 1;
  int64_t nextWakeup = INT64_MAX;
  rasSocksHandleTimeouts(100 * CLOCK_UNITS_PER_SEC, &nextWakeup);
  EXPECT_EQ(4, g_socketCloseCalls);
  EXPECT_EQ(nullptr, rasSocketsHead);
  ASSERT_EQ(2u, g_events.size());
  EXPECT_EQ("PEER_INIT_TIMEOUT", g_events[0].type);
  EXPECT_EQ("PEER_INIT_TIMEOUT", g_events[1].type);
}

TEST_F(RasNetMicrotest, SocketTerminateFiltersRetryQueueAndStartsRetryClock) {
  rasConnection* conn = MakeConn();
  rasSocket* sock = MakeSock(conn, RAS_SOCK_READY, 0);
  sock->lastSendTime = g_clockNano - 1;
  sock->lastRecvTime = g_clockNano - 2;
  MakeQueuedMsg(conn, RAS_MSG_CONNINIT, 1, 3);
  MakeQueuedMsg(conn, RAS_MSG_KEEPALIVE, 2, 4);
  rasMsgMeta* kept = MakeQueuedMsg(conn, RAS_MSG_COLLREQ, 3, 5);
  rasSocketTerminate(sock, false, 17, true);
  EXPECT_EQ(nullptr, conn->sock);
  EXPECT_EQ(g_clockNano - 17, conn->startRetryTime);
  EXPECT_EQ(g_clockNano - 17, conn->lastRetryTime);
  ASSERT_EQ(kept, ncclIntruQueueHead(&conn->sendQ));
  EXPECT_EQ(0, kept->offset);
  EXPECT_EQ(1, g_socketShutdownCalls);
  EXPECT_EQ(SHUT_WR, g_lastShutdownHow);
  EXPECT_EQ(RAS_SOCK_TERMINATING, sock->status);
  EXPECT_EQ(POLLIN, rasPfds[0].events);
  EXPECT_EQ(conn, g_purgedConnections.back());
}

TEST_F(RasNetMicrotest, SocketTerminateSkipsRetryForIdleUnlinkedConnection) {
  rasConnection* conn = MakeConn();
  rasSocket* sock = MakeSock(conn, RAS_SOCK_READY, 0);
  sock->lastSendTime = sock->lastRecvTime = g_clockNano - RAS_IDLE_TIMEOUT;
  rasSocketTerminate(sock, true, 0, true);
  EXPECT_EQ(0, conn->startRetryTime);
  EXPECT_EQ(nullptr, rasSocketsHead);
  EXPECT_EQ(1, g_socketCloseCalls);
}

TEST_F(RasNetMicrotest, SocketTerminateRetriesLinkConnectionEvenWhenRetryDisabled) {
  rasConnection* conn = MakeConn();
  rasSocket* sock = MakeSock(conn, RAS_SOCK_CONNECTING, 0);
  AddLinkConn(&rasNextLink, conn, 1);
  rasSocketTerminate(sock, true, 9, false);
  EXPECT_EQ(g_clockNano - 9, conn->startRetryTime);
}

TEST_F(RasNetMicrotest, SocketTerminateFinalizesAndClearsPollEntry) {
  rasSocket* sock = MakeSock(nullptr, RAS_SOCK_CLOSED, 3);
  sock->recvMsg = MakeRecvMsg(RAS_MSG_KEEPALIVE);
  rasSocketTerminate(sock, true);
  EXPECT_EQ(nullptr, rasSocketsHead);
  EXPECT_EQ(NCCL_INVALID_SOCKET, rasPfds[3].fd);
  EXPECT_EQ(0, rasPfds[3].events);
  EXPECT_EQ(1, g_socketCloseCalls);
}

TEST_F(RasNetMicrotest, SocketTerminateAlreadyTerminatingWaitsForReceive) {
  rasConnection* conn = MakeConn();
  rasSocket* sock = MakeSock(conn, RAS_SOCK_TERMINATING, 0);
  conn->sock = nullptr;
  rasSocketTerminate(sock, false);
  EXPECT_EQ(0, g_socketShutdownCalls);
  EXPECT_EQ(sock, rasSocketsHead);
}

TEST_F(RasNetMicrotest, ConnectingEventLoopAdvancesConnectSideHandshake) {
  rasConnection* conn = MakeConn();
  rasSocket* sock = MakeSock(conn, RAS_SOCK_CONNECTING, 0);
  rasPfds[0].events = POLLOUT;
  rasSockEventLoop(sock, 0);
  EXPECT_EQ(RAS_SOCK_HANDSHAKE, sock->status);
  EXPECT_EQ(g_clockNano, sock->lastSendTime);
  ASSERT_FALSE(ncclIntruQueueEmpty(&conn->sendQ));
  EXPECT_EQ(RAS_MSG_CONNINIT, ncclIntruQueueHead(&conn->sendQ)->msg.type);
}

TEST_F(RasNetMicrotest, ConnectingEventLoopHandlesReadyFailuresAndInvalidSides) {
  rasSocket* failed = MakeSock(nullptr, RAS_SOCK_CONNECTING, 0);
  g_socketReadyResult = ncclSystemError;
  rasSockEventLoop(failed, 0);
  EXPECT_EQ(nullptr, rasSocketsHead);

  g_socketReadyResult = ncclSuccess;
  rasSocket* noConn = MakeSock(nullptr, RAS_SOCK_CONNECTING, 1);
  rasPfds[1].events = POLLOUT;
  rasSockEventLoop(noConn, 1);
  EXPECT_EQ(nullptr, rasSocketsHead);

  rasConnection* conn = MakeConn();
  rasSocket* current = MakeSock(conn, RAS_SOCK_CONNECTING, 2);
  rasSocket* stale = MakeSock(conn, RAS_SOCK_CONNECTING, 3);
  conn->sock = current;
  rasPfds[3].events = POLLOUT;
  rasSockEventLoop(stale, 3);
  EXPECT_EQ(current, conn->sock);
  EXPECT_EQ(current, rasSocketsHead);
}

TEST_F(RasNetMicrotest, ConnectingEventLoopTerminatesWhenConnInitAllocationFails) {
  rasConnection* conn = MakeConn();
  rasSocket* sock = MakeSock(conn, RAS_SOCK_CONNECTING, 0);
  rasPfds[0].events = POLLOUT;
  g_msgAllocResult = ncclSystemError;
  rasSockEventLoop(sock, 0);
  EXPECT_EQ(nullptr, conn->sock);
  EXPECT_EQ(nullptr, rasSocketsHead);
}

TEST_F(RasNetMicrotest, ConnectingEventLoopHandlesNotReadyStates) {
  rasSocket* connecting = MakeSock(nullptr, RAS_SOCK_CONNECTING, 0);
  g_socketReadyValue = 0;
  g_socketReadyState = ncclSocketStateConnecting;
  rasSockEventLoop(connecting, 0);
  EXPECT_EQ(POLL_FD_IGNORE, rasPfds[0].fd);

  rasSocket* badHandshake = MakeSock(nullptr, RAS_SOCK_CONNECTING, 1);
  badHandshake->sock.socketDescriptor = NCCL_INVALID_SOCKET;
  g_socketReadyState = ncclSocketStateBadHandshake;
  rasSockEventLoop(badHandshake, 1);
  EXPECT_EQ(connecting, rasSocketsHead);
  EXPECT_EQ(connecting, rasSocketsTail);
}

TEST_F(RasNetMicrotest, EventLoopSendSuccessAndFailurePaths) {
  rasConnection* conn = MakeConn();
  rasSocket* sock = MakeSock(conn, RAS_SOCK_READY, 0);
  rasPfds[0].revents = POLLOUT;
  g_sendMsgAllSent = true;
  rasSockEventLoop(sock, 0);
  EXPECT_EQ(g_clockNano, sock->lastSendTime);
  EXPECT_EQ(POLLIN, rasPfds[0].events);

  rasPfds[0].events |= POLLOUT;
  rasPfds[0].revents = POLLOUT;
  g_sendMsgAllSent = false;
  g_sendMsgClosed = 1;
  rasSockEventLoop(sock, 0);
  EXPECT_EQ(nullptr, conn->sock);
  ASSERT_FALSE(g_events.empty());
  EXPECT_EQ("PEER_DISCONNECTED", g_events.back().type);
}

TEST_F(RasNetMicrotest, EventLoopRejectsMissingStaleAndFailedSendConnections) {
  rasSocket* noConn = MakeSock(nullptr, RAS_SOCK_READY, 0);
  rasPfds[0].revents = POLLOUT;
  rasSockEventLoop(noConn, 0);
  EXPECT_EQ(nullptr, rasSocketsHead);

  rasConnection* conn = MakeConn();
  rasSocket* current = MakeSock(conn, RAS_SOCK_READY, 1);
  rasSocket* stale = MakeSock(conn, RAS_SOCK_READY, 2);
  conn->sock = current;
  rasPfds[2].revents = POLLOUT;
  rasSockEventLoop(stale, 2);
  EXPECT_EQ(current, conn->sock);

  rasPfds[1].revents = POLLOUT;
  g_sendMsgResult = ncclSystemError;
  rasSockEventLoop(current, 1);
  EXPECT_EQ(nullptr, conn->sock);
}

TEST_F(RasNetMicrotest, EventLoopReceivesMessagesAndResumesConnection) {
  rasConnection* conn = MakeConn();
  rasSocket* sock = MakeSock(conn, RAS_SOCK_READY, 0);
  conn->experiencingDelays = true;
  conn->startRetryTime = 1;
  rasPfds[0].revents = POLLIN;
  g_recvSteps.push_back({ncclSuccess, 0, MakeRecvMsg(RAS_MSG_KEEPALIVE)});
  rasSockEventLoop(sock, 0);
  EXPECT_EQ(1, g_handleMsgCalls);
  EXPECT_EQ(RAS_MSG_KEEPALIVE, g_lastHandledType);
  EXPECT_FALSE(conn->experiencingDelays);
  EXPECT_EQ(0, conn->startRetryTime);
  EXPECT_EQ(g_clockNano, sock->lastRecvTime);
  ASSERT_FALSE(g_events.empty());
  EXPECT_EQ("PEER_RECOVERED", g_events.back().type);
}

TEST_F(RasNetMicrotest, EventLoopReceiveFailureAndClosedSocketFinalize) {
  rasSocket* failed = MakeSock(nullptr, RAS_SOCK_HANDSHAKE, 0);
  rasPfds[0].revents = POLLIN;
  g_recvSteps.push_back({ncclSystemError, 0, nullptr});
  rasSockEventLoop(failed, 0);
  EXPECT_EQ(nullptr, rasSocketsHead);

  rasConnection* conn = MakeConn();
  rasSocket* closed = MakeSock(conn, RAS_SOCK_HANDSHAKE, 1);
  rasPfds[1].revents = POLLIN;
  g_recvSteps.push_back({ncclSuccess, 1, nullptr});
  rasSockEventLoop(closed, 1);
  EXPECT_EQ(nullptr, conn->sock);
  ASSERT_FALSE(g_events.empty());
  EXPECT_NE(std::string::npos, g_events.back().details.find("new"));
}

TEST_F(RasNetMicrotest, EventLoopReportsCurrentSocketClosedOnReceive) {
  rasConnection* conn = MakeConn();
  rasSocket* sock = MakeSock(conn, RAS_SOCK_READY, 0);
  rasPfds[0].revents = POLLIN;
  g_recvSteps.push_back({ncclSuccess, 1, nullptr});
  rasSockEventLoop(sock, 0);
  ASSERT_FALSE(g_events.empty());
  EXPECT_NE(std::string::npos, g_events.back().details.find("current"));
}

TEST_F(RasNetMicrotest, EventLoopStopsReceiveLoopAfterHandlerTerminatesPollEntry) {
  rasSocket* sock = MakeSock(nullptr, RAS_SOCK_HANDSHAKE, 0);
  rasPfds[0].revents = POLLIN;
  g_clearPollInOnHandle = true;
  g_recvSteps.push_back({ncclSuccess, 0, MakeRecvMsg(RAS_MSG_KEEPALIVE)});
  g_recvSteps.push_back({ncclSuccess, 0, MakeRecvMsg(RAS_MSG_KEEPALIVE)});
  rasSockEventLoop(sock, 0);
  EXPECT_EQ(1, g_handleMsgCalls);
  EXPECT_EQ(1u, g_recvSteps.size());
}

TEST_F(RasNetMicrotest, ConnectionTimeoutRetriesConnectingSocket) {
  rasConnection* conn = MakeConn();
  rasSocket* sock = MakeSock(conn, RAS_SOCK_CONNECTING, 0);
  sock->sock.state = ncclSocketStateConnecting;
  sock->lastSendTime = g_clockNano - RAS_CONNECT_RETRY - 1;
  g_socketReadyValue = 0;
  g_socketReadyState = ncclSocketStateConnecting;
  int64_t nextWakeup = INT64_MAX;
  rasConnsHandleTimeouts(g_clockNano, &nextWakeup);
  EXPECT_EQ(g_clockNano, sock->lastSendTime);
  EXPECT_EQ(g_clockNano + RAS_CONNECT_RETRY, nextWakeup);

  g_socketReadyValue = 1;
  g_socketReadyState = ncclSocketStateReady;
  g_clockNano += RAS_CONNECT_RETRY + 1;
  rasConnsHandleTimeouts(g_clockNano, &nextWakeup);
  EXPECT_EQ(sock->sock.socketDescriptor, rasPfds[0].fd);
}

TEST_F(RasNetMicrotest, ConnectionTimeoutSchedulesConnectingRetryBoundary) {
  rasConnection* conn = MakeConn();
  rasSocket* sock = MakeSock(conn, RAS_SOCK_CONNECTING, 0);
  sock->sock.state = ncclSocketStateConnecting;
  sock->lastSendTime = g_clockNano;
  int64_t nextWakeup = INT64_MAX;
  rasConnsHandleTimeouts(g_clockNano, &nextWakeup);
  EXPECT_EQ(g_clockNano + RAS_CONNECT_RETRY, nextWakeup);
  EXPECT_EQ(0, g_socketReadyCalls);
}

TEST_F(RasNetMicrotest, ConnectionTimeoutTerminatesReadyFailureAndStuckSend) {
  rasConnection* conn = MakeConn();
  rasSocket* sock = MakeSock(conn, RAS_SOCK_CONNECTING, 0);
  sock->sock.state = ncclSocketStateConnecting;
  sock->lastSendTime = 1;
  g_socketReadyResult = ncclSystemError;
  int64_t nextWakeup = INT64_MAX;
  rasConnsHandleTimeouts(g_clockNano, &nextWakeup);
  EXPECT_EQ(nullptr, conn->sock);

  g_socketReadyResult = ncclSuccess;
  rasSocket* replacement = MakeSock(conn, RAS_SOCK_READY, 1);
  replacement->lastSendTime = 1;
  MakeQueuedMsg(conn, RAS_MSG_COLLREQ, 1);
  rasConnsHandleTimeouts(g_clockNano, &nextWakeup);
  EXPECT_EQ(nullptr, conn->sock);
  ASSERT_FALSE(g_events.empty());
  EXPECT_EQ("PEER_SEND_STUCK", g_events.back().type);
}

TEST_F(RasNetMicrotest, ConnectionTimeoutSchedulesPendingSend) {
  rasConnection* conn = MakeConn();
  rasSocket* sock = MakeSock(conn, RAS_SOCK_READY, 0);
  sock->lastSendTime = 10;
  rasMsgMeta* meta = MakeQueuedMsg(conn, RAS_MSG_COLLREQ, 20);
  int64_t nextWakeup = INT64_MAX;
  rasConnsHandleTimeouts(21, &nextWakeup);
  EXPECT_EQ(std::max(sock->lastSendTime, meta->enqueueTime) + RAS_STUCK_TIMEOUT, nextWakeup);
}

TEST_F(RasNetMicrotest, ConnectionTimeoutDeclaresPeerDead) {
  rasConnection* conn = MakeConn(5555);
  conn->startRetryTime = 1;
  int64_t nextWakeup = INT64_MAX;
  rasConnsHandleTimeouts(RAS_PEER_DEAD_TIMEOUT + 2, &nextWakeup);
  EXPECT_EQ(1, g_collReqInitCalls);
  EXPECT_EQ(1, g_sendCollReqCalls);
  EXPECT_EQ(RAS_BC_DEADPEER, g_lastCollReq.type);
  EXPECT_EQ(htons(5555), g_lastCollReq.deadPeer.addr.sin.sin_port);
  ASSERT_FALSE(g_events.empty());
  EXPECT_EQ("PEER_TIMEOUT_DEAD", g_events.back().type);
}

TEST_F(RasNetMicrotest, ConnectionTimeoutWarnsOnceAndRetriesMissingSocket) {
  rasConnection* conn = MakeConn();
  conn->startRetryTime = g_clockNano - RAS_CONNECT_WARN - 1;
  conn->lastRetryTime = g_clockNano - RAS_CONNECT_RETRY - 1;
  int64_t nextWakeup = INT64_MAX;
  rasConnsHandleTimeouts(g_clockNano, &nextWakeup);
  EXPECT_TRUE(conn->experiencingDelays);
  EXPECT_EQ(1u, g_purgedConnections.size());
  EXPECT_EQ(1, g_socketInitCalls);
  ASSERT_FALSE(g_events.empty());
  EXPECT_EQ("PEER_UNRESPONSIVE", g_events[0].type);
}

TEST_F(RasNetMicrotest, ConnectionTimeoutReportsFirstRetry) {
  rasConnection* conn = MakeConn();
  conn->startRetryTime = g_clockNano - RAS_CONNECT_RETRY - 1;
  conn->lastRetryTime = 0;
  int64_t nextWakeup = INT64_MAX;
  rasConnsHandleTimeouts(g_clockNano, &nextWakeup);
  ASSERT_FALSE(g_events.empty());
  EXPECT_EQ("PEER_RETRY", g_events[0].type);
  EXPECT_EQ(1, g_socketInitCalls);
}

TEST_F(RasNetMicrotest, ConnectionTimeoutSchedulesWarningAndRetryBoundaries) {
  rasConnection* conn = MakeConn();
  conn->startRetryTime = 10;
  conn->lastRetryTime = 20;
  int64_t nextWakeup = INT64_MAX;
  rasConnsHandleTimeouts(21, &nextWakeup);
  EXPECT_EQ(20 + RAS_CONNECT_RETRY, nextWakeup);
  EXPECT_EQ(0, g_socketInitCalls);
}

TEST_F(RasNetMicrotest, ConnectionDisconnectDropsLinksAndTerminates) {
  rasConnection* conn = MakeConn(1001);
  MakeSock(conn, RAS_SOCK_READY, 0);
  AddLinkConn(&rasNextLink, conn, 1);
  AddLinkConn(&rasPrevLink, conn, 1);
  ncclSocketAddress addr = conn->addr;
  rasConnDisconnect(&addr);
  EXPECT_EQ(nullptr, rasConnsHead);
  EXPECT_EQ(nullptr, rasSocketsHead);
  EXPECT_EQ(nullptr, rasNextLink.conns->conn);
  EXPECT_EQ(nullptr, rasPrevLink.conns->conn);
}

TEST_F(RasNetMicrotest, NetworkTimeoutTerminatesUnlinkedIdleConnectionOnly) {
  rasConnection* idle = MakeConn(1001);
  rasConnection* queued = MakeConn(1002);
  MakeQueuedMsg(queued, RAS_MSG_COLLREQ, 1);
  rasConnection* linked = MakeConn(1003);
  AddLinkConn(&rasNextLink, linked, 0);
  int64_t nextWakeup = INT64_MAX;
  rasNetHandleTimeouts(g_clockNano, &nextWakeup);
  EXPECT_EQ(queued, rasConnsHead);
  EXPECT_EQ(linked, rasConnsTail);
  EXPECT_TRUE(linked->linkFlag);
  EXPECT_FALSE(queued->linkFlag);
  (void)idle;
}

TEST_F(RasNetMicrotest, LinkTimeoutCreatesMissingPrimaryAfterWarning) {
  SetPeers({1001, 1002});
  AddLinkConn(&rasNextLink, nullptr, 1);
  rasNextLink.lastUpdatePeersTime = 1;
  int64_t nextWakeup = INT64_MAX;
  ASSERT_EQ(ncclSuccess, rasLinkHandleNetTimeouts(&rasNextLink, RAS_CONNECT_WARN + 2, &nextWakeup));
  ASSERT_NE(nullptr, rasNextLink.conns->conn);
  EXPECT_TRUE(rasNextLink.conns->conn->linkFlag);
  EXPECT_EQ(0, rasNextLink.lastUpdatePeersTime);
}

TEST_F(RasNetMicrotest, LinkTimeoutSchedulesMissingPrimaryBeforeWarning) {
  SetPeers({1001});
  AddLinkConn(&rasNextLink, nullptr, 0);
  rasNextLink.lastUpdatePeersTime = 10;
  int64_t nextWakeup = INT64_MAX;
  EXPECT_EQ(ncclSuccess, rasLinkHandleNetTimeouts(&rasNextLink, 11, &nextWakeup));
  EXPECT_EQ(10 + RAS_CONNECT_WARN, nextWakeup);
}

TEST_F(RasNetMicrotest, LinkedConnectionSendsKeepAliveAndSchedulesTimeouts) {
  rasConnection* conn = MakeConn();
  rasSocket* sock = MakeSock(conn, RAS_SOCK_READY, 0);
  sock->lastSendTime = g_clockNano - RAS_KEEPALIVE_INTERVAL - 1;
  sock->lastRecvTime = g_clockNano;
  AddLinkConn(&rasNextLink, conn, 0);
  int64_t nextWakeup = INT64_MAX;
  rasNetHandleTimeouts(g_clockNano, &nextWakeup);
  ASSERT_FALSE(ncclIntruQueueEmpty(&conn->sendQ));
  EXPECT_EQ(RAS_MSG_KEEPALIVE, ncclIntruQueueHead(&conn->sendQ)->msg.type);
  EXPECT_EQ(g_clockNano + RAS_KEEPALIVE_TIMEOUT_WARN, nextWakeup);
}

TEST_F(RasNetMicrotest, LinkedConnectionWarnsThenTerminatesOnKeepAliveTimeout) {
  rasConnection* conn = MakeConn();
  rasSocket* sock = MakeSock(conn, RAS_SOCK_READY, 0);
  sock->lastSendTime = g_clockNano;
  sock->lastRecvTime = g_clockNano - RAS_KEEPALIVE_TIMEOUT_ERROR - 1;
  AddLinkConn(&rasNextLink, conn, 0);
  int64_t nextWakeup = INT64_MAX;
  rasNetHandleTimeouts(g_clockNano, &nextWakeup);
  EXPECT_TRUE(conn->experiencingDelays);
  EXPECT_EQ(nullptr, conn->sock);
  EXPECT_EQ(g_clockNano, nextWakeup);
  ASSERT_GE(g_events.size(), 2u);
  EXPECT_EQ("PEER_UNRESPONSIVE", g_events[0].type);
  EXPECT_EQ("PEER_KEEPALIVE_TIMEOUT", g_events[1].type);
}

TEST_F(RasNetMicrotest, KeepAliveEncodesLinkMaskHashesAndTimestamp) {
  rasConnection* conn = MakeConn();
  AddLinkConn(&rasNextLink, conn, 0);
  AddLinkConn(&rasPrevLink, conn, 0, true);
  rasConnSendKeepAlive(conn, true);
  rasMsgMeta* meta = ncclIntruQueueHead(&conn->sendQ);
  ASSERT_NE(nullptr, meta);
  EXPECT_EQ(RAS_MSG_KEEPALIVE, meta->msg.type);
  EXPECT_EQ(2, meta->msg.keepAlive.linkMask);
  EXPECT_EQ(1, meta->msg.keepAlive.nack);
  EXPECT_EQ(rasPeersHash, meta->msg.keepAlive.peersHash);
  EXPECT_EQ(rasDeadPeersHash, meta->msg.keepAlive.deadPeersHash);
  EXPECT_EQ(g_realtime.tv_sec, meta->msg.keepAlive.realTime.tv_sec);
}

TEST_F(RasNetMicrotest, KeepAliveEncodesBothNonExternalLinks) {
  rasConnection* conn = MakeConn();
  AddLinkConn(&rasNextLink, conn, 0);
  AddLinkConn(&rasPrevLink, conn, 0);
  rasConnSendKeepAlive(conn);
  ASSERT_FALSE(ncclIntruQueueEmpty(&conn->sendQ));
  EXPECT_EQ(3, ncclIntruQueueHead(&conn->sendQ)->msg.keepAlive.linkMask);
}

TEST_F(RasNetMicrotest, KeepAliveAllocationFailureIsIgnored) {
  rasConnection* conn = MakeConn();
  g_msgAllocResult = ncclSystemError;
  rasConnSendKeepAlive(conn);
  EXPECT_TRUE(ncclIntruQueueEmpty(&conn->sendQ));
}

TEST_F(RasNetMicrotest, KeepAliveHandlerRequiresConnection) {
  rasSocket* sock = MakeSock(nullptr, RAS_SOCK_READY, 0);
  rasMsg msg{};
  msg.type = RAS_MSG_KEEPALIVE;
  EXPECT_EQ(ncclInternalError, rasMsgHandleKeepAlive(&msg, sock));
}

TEST_F(RasNetMicrotest, KeepAliveHandlerUpdatesLinksStatsAndHashes) {
  SetPeers({1001});
  rasConnection* conn = MakeConn(1001);
  rasSocket* sock = MakeSock(conn, RAS_SOCK_READY, 0);
  g_peerFindResult = 0;
  g_realtime = {20, 500};
  rasMsg msg{};
  msg.type = RAS_MSG_KEEPALIVE;
  msg.keepAlive.realTime = {19, 400};
  msg.keepAlive.peersHash = rasPeersHash;
  msg.keepAlive.deadPeersHash = rasDeadPeersHash;
  msg.keepAlive.linkMask = 3;
  ASSERT_EQ(ncclSuccess, rasMsgHandleKeepAlive(&msg, sock));
  EXPECT_EQ(rasPeersHash, conn->lastRecvPeersHash);
  EXPECT_EQ(rasDeadPeersHash, conn->lastRecvDeadPeersHash);
  EXPECT_EQ(1'000'000'100, conn->travelTimeMin);
  EXPECT_EQ(1'000'000'100, conn->travelTimeMax);
  EXPECT_EQ(1, conn->travelTimeCount);
  ASSERT_NE(nullptr, rasNextLink.conns);
  ASSERT_NE(nullptr, rasPrevLink.conns);
  EXPECT_EQ(conn, rasNextLink.conns->conn);
  EXPECT_EQ(conn, rasPrevLink.conns->conn);
}

TEST_F(RasNetMicrotest, KeepAliveHandlerNacksUnneededConnection) {
  rasConnection* conn = MakeConn();
  rasSocket* sock = MakeSock(conn, RAS_SOCK_READY, 0);
  rasMsg msg{};
  msg.type = RAS_MSG_KEEPALIVE;
  msg.keepAlive.realTime = g_realtime;
  msg.keepAlive.peersHash = rasPeersHash;
  msg.keepAlive.deadPeersHash = rasDeadPeersHash;
  ASSERT_EQ(ncclSuccess, rasMsgHandleKeepAlive(&msg, sock));
  ASSERT_FALSE(ncclIntruQueueEmpty(&conn->sendQ));
  EXPECT_EQ(1, ncclIntruQueueHead(&conn->sendQ)->msg.keepAlive.nack);
}

TEST_F(RasNetMicrotest, KeepAliveHandlerRequestsPeerUpdateOnHashMismatch) {
  rasConnection* conn = MakeConn();
  rasSocket* sock = MakeSock(conn, RAS_SOCK_READY, 0);
  rasMsg msg{};
  msg.type = RAS_MSG_KEEPALIVE;
  msg.keepAlive.realTime = g_realtime;
  msg.keepAlive.nack = 1;
  msg.keepAlive.peersHash = 8;
  msg.keepAlive.deadPeersHash = 9;
  EXPECT_EQ(ncclSuccess, rasMsgHandleKeepAlive(&msg, sock));
  EXPECT_EQ(1, g_sendPeersUpdateCalls);
  EXPECT_EQ(conn, g_lastPeersUpdateConn);
  g_sendPeersUpdateResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, rasMsgHandleKeepAlive(&msg, sock));
}

TEST_F(RasNetMicrotest, LinkEntryAddFindUpdateAndDrop) {
  rasConnection* first = MakeConn(1001);
  rasConnection* second = MakeConn(1002);
  rasLinkConn* entry = nullptr;
  int index = -1;
  ASSERT_EQ(ncclSuccess, rasLinkConnAdd(&rasNextLink, first, 1, false, &index, &entry));
  EXPECT_EQ(0, index);
  EXPECT_EQ(first, entry->conn);
  EXPECT_EQ(entry, rasLinkConnFind(&rasNextLink, first, &index));
  EXPECT_EQ(0, index);
  entry->external = true;
  EXPECT_EQ(ncclSuccess, rasLinkConnAdd(&rasNextLink, first, 1));
  EXPECT_FALSE(entry->external);
  EXPECT_EQ(ncclSuccess, rasLinkConnAdd(&rasNextLink, second, 2));
  EXPECT_EQ((std::vector<int>{1, 2}), LinkPeerIndices(rasNextLink));
  rasLinkConnDrop(&rasNextLink, second);
  EXPECT_EQ((std::vector<int>{1}), LinkPeerIndices(rasNextLink));
  rasLinkConnDrop(&rasNextLink, first);
  EXPECT_EQ(-1, rasNextLink.conns->peerIdx);
  EXPECT_EQ(nullptr, rasNextLink.conns->conn);
  EXPECT_EQ(nullptr, rasLinkConnFind(&rasNextLink, first, &index));
  EXPECT_EQ(-1, index);
}

TEST_F(RasNetMicrotest, LinkEntryAddRejectsInvalidAndMismatchedInputs) {
  rasConnection* first = MakeConn(1001);
  rasConnection* second = MakeConn(1002);
  EXPECT_EQ(ncclInternalError, rasLinkConnAdd(&rasNextLink, first, -1));
  AddLinkConn(&rasNextLink, first, 1);
  EXPECT_EQ(ncclInternalError, rasLinkConnAdd(&rasNextLink, first, 2));
  EXPECT_EQ(ncclInternalError, rasLinkConnAdd(&rasNextLink, second, 1));
  EXPECT_EQ(ncclInternalError, rasLinkConnUpdate(&rasNextLink, nullptr, 1));
  EXPECT_EQ(ncclInternalError, rasLinkConnUpdate(&rasNextLink, first, -1));
}

TEST_F(RasNetMicrotest, LinkEntryPretendAndNoInsertLeaveListUntouched) {
  rasConnection* conn = MakeConn();
  int index = -1;
  EXPECT_EQ(ncclSuccess, rasLinkConnAdd(&rasNextLink, conn, 3, true, &index));
  EXPECT_EQ(0, index);
  EXPECT_EQ(nullptr, rasNextLink.conns);
  EXPECT_EQ(ncclSuccess, rasLinkConnUpdate(&rasNextLink, conn, 3));
  EXPECT_EQ(nullptr, rasNextLink.conns);
}

TEST_F(RasNetMicrotest, LinkEntryInsertionCoversMonotonicRolloverAndUnknownOrdering) {
  rasConnection* peerFive = MakeConn(1005);
  rasConnection* peerSeven = MakeConn(1007);
  rasConnection* peerSix = MakeConn(1006);
  ASSERT_EQ(ncclSuccess, rasLinkConnAdd(&rasNextLink, peerFive, 5));
  ASSERT_EQ(ncclSuccess, rasLinkConnAdd(&rasNextLink, peerSeven, 7));
  ASSERT_EQ(ncclSuccess, rasLinkConnAdd(&rasNextLink, peerSix, 6));
  EXPECT_EQ((std::vector<int>{5, 6, 7}), LinkPeerIndices(rasNextLink));

  FreeLink(&rasNextLink);
  AddLinkConn(&rasNextLink, peerSeven, 8);
  AddLinkConn(&rasNextLink, peerFive, 1);
  rasConnection* peerNine = MakeConn(1009);
  ASSERT_EQ(ncclSuccess, rasLinkConnAdd(&rasNextLink, peerNine, 9));
  EXPECT_EQ((std::vector<int>{8, 9, 1}), LinkPeerIndices(rasNextLink));

  AddLinkConn(&rasNextLink, nullptr, -1);
  rasConnection* peerTwo = MakeConn(1002);
  ASSERT_EQ(ncclSuccess, rasLinkConnAdd(&rasNextLink, peerTwo, 2));
  EXPECT_EQ((std::vector<int>{8, 9, 1, 2, -1}), LinkPeerIndices(rasNextLink));
}

TEST_F(RasNetMicrotest, LinkEntryMovesUnknownConnectionIntoSortedPosition) {
  rasConnection* first = MakeConn(1001);
  rasConnection* third = MakeConn(1003);
  rasConnection* moved = MakeConn(1002);
  AddLinkConn(&rasNextLink, first, 1);
  AddLinkConn(&rasNextLink, third, 3);
  AddLinkConn(&rasNextLink, moved, -1, true);
  ASSERT_EQ(ncclSuccess, rasLinkConnAdd(&rasNextLink, moved, 2));
  EXPECT_EQ((std::vector<int>{1, 2, 3}), LinkPeerIndices(rasNextLink));
  EXPECT_FALSE(rasNextLink.conns->next->external);
}

TEST_F(RasNetMicrotest, LinkEntryRejectsMovingUnknownConnectionLater) {
  rasConnection* moved = MakeConn(1002);
  rasConnection* first = MakeConn(1001);
  AddLinkConn(&rasNextLink, moved, -1, true);
  AddLinkConn(&rasNextLink, first, 1);
  EXPECT_EQ(ncclInternalError, rasLinkConnAdd(&rasNextLink, moved, 2));
}

TEST_F(RasNetMicrotest, LinkEntryFillsPrimaryPlaceholderAndSanitizesFallbacks) {
  rasConnection* primary = MakeConn(1001);
  MakeSock(primary, RAS_SOCK_READY, 0);
  rasConnection* fallback = MakeConn(1002);
  AddLinkConn(&rasNextLink, nullptr, 1);
  AddLinkConn(&rasNextLink, fallback, 2, true);
  rasNextLink.lastUpdatePeersTime = 99;
  ASSERT_EQ(ncclSuccess, rasLinkConnAdd(&rasNextLink, primary, 1));
  EXPECT_EQ(primary, rasNextLink.conns->conn);
  EXPECT_EQ(nullptr, rasNextLink.conns->next);
  EXPECT_EQ(0, rasNextLink.lastUpdatePeersTime);
}

TEST_F(RasNetMicrotest, ExternalLinkEntriesAddMoveAndDrop) {
  rasConnection* primary = MakeConn(1001);
  rasConnection* external = MakeConn(1002);
  ASSERT_EQ(ncclSuccess, rasLinkConnAddExternal(&rasNextLink, primary, 1));
  EXPECT_FALSE(rasNextLink.conns->external);
  ASSERT_EQ(ncclSuccess, rasLinkConnAddExternal(&rasNextLink, external, -1));
  EXPECT_TRUE(rasNextLink.conns->next->external);
  rasLinkConnDrop(&rasNextLink, external, true);
  EXPECT_EQ((std::vector<int>{1}), LinkPeerIndices(rasNextLink));
  ASSERT_EQ(ncclSuccess, rasLinkConnAddExternal(&rasNextLink, external, -1));
  ASSERT_EQ(ncclSuccess, rasLinkConnAddExternal(&rasNextLink, external, 2));
  EXPECT_EQ((std::vector<int>{1, 2}), LinkPeerIndices(rasNextLink));
  rasLinkConnDrop(&rasNextLink, external, true);
  EXPECT_EQ((std::vector<int>{1, 2}), LinkPeerIndices(rasNextLink));
  rasLinkConnDrop(&rasNextLink, external);
  EXPECT_EQ((std::vector<int>{1}), LinkPeerIndices(rasNextLink));
  rasLinkConnDrop(&rasNextLink, primary, true);
  EXPECT_EQ(primary, rasNextLink.conns->conn);
}

TEST_F(RasNetMicrotest, ExternalLinkEntriesRejectNullAndMismatch) {
  rasConnection* first = MakeConn(1001);
  rasConnection* second = MakeConn(1002);
  EXPECT_EQ(ncclInternalError, rasLinkConnAddExternal(&rasNextLink, nullptr, 0));
  AddLinkConn(&rasNextLink, first, 1);
  EXPECT_EQ(ncclInternalError, rasLinkConnAddExternal(&rasNextLink, first, 2));
  EXPECT_EQ(ncclInternalError, rasLinkConnAddExternal(&rasNextLink, second, 1));
}

TEST_F(RasNetMicrotest, ExternalLinkEntryFillsPlaceholderAndSanitizes) {
  rasConnection* conn = MakeConn(1001);
  MakeSock(conn, RAS_SOCK_READY, 0);
  AddLinkConn(&rasNextLink, nullptr, 1);
  AddLinkConn(&rasNextLink, MakeConn(1002), 2, true);
  ASSERT_EQ(ncclSuccess, rasLinkConnAddExternal(&rasNextLink, conn, 1));
  EXPECT_EQ(conn, rasNextLink.conns->conn);
  EXPECT_EQ(nullptr, rasNextLink.conns->next);
  EXPECT_EQ(ncclSuccess, rasLinkConnAddExternal(&rasNextLink, conn, 1));
}

TEST_F(RasNetMicrotest, ExternalLinkEntryOrderingAndMoveMatchInternalOrdering) {
  rasConnection* first = MakeConn(1001);
  rasConnection* third = MakeConn(1003);
  rasConnection* moved = MakeConn(1002);
  AddLinkConn(&rasNextLink, first, 1);
  AddLinkConn(&rasNextLink, third, 3);
  AddLinkConn(&rasNextLink, moved, -1, true);
  ASSERT_EQ(ncclSuccess, rasLinkConnAddExternal(&rasNextLink, moved, 2));
  EXPECT_EQ((std::vector<int>{1, 2, 3}), LinkPeerIndices(rasNextLink));

  FreeLink(&rasNextLink);
  AddLinkConn(&rasNextLink, third, 8);
  AddLinkConn(&rasNextLink, first, 1);
  rasConnection* peerNine = MakeConn(1009);
  ASSERT_EQ(ncclSuccess, rasLinkConnAddExternal(&rasNextLink, peerNine, 9));
  EXPECT_EQ((std::vector<int>{8, 9, 1}), LinkPeerIndices(rasNextLink));
}

TEST_F(RasNetMicrotest, ExternalLinkEntryRejectsMovingUnknownConnectionLater) {
  rasConnection* moved = MakeConn(1002);
  rasConnection* first = MakeConn(1001);
  AddLinkConn(&rasNextLink, moved, -1, true);
  AddLinkConn(&rasNextLink, first, 1);
  EXPECT_EQ(ncclInternalError, rasLinkConnAddExternal(&rasNextLink, moved, 2));
}

TEST_F(RasNetMicrotest, DroppingPrimaryPromotesFallback) {
  rasConnection* primary = MakeConn(1001);
  rasConnection* fallback = MakeConn(1002);
  AddLinkConn(&rasNextLink, primary, 1);
  AddLinkConn(&rasNextLink, fallback, 2, true);
  rasLinkConnDrop(&rasNextLink, primary);
  EXPECT_EQ(fallback, rasNextLink.conns->conn);
  EXPECT_FALSE(rasNextLink.conns->external);
}

TEST_F(RasNetMicrotest, ConnectionResumeCleansFallbacksAndEnablesSend) {
  rasConnection* conn = MakeConn(1001);
  rasSocket* sock = MakeSock(conn, RAS_SOCK_READY, 0);
  conn->startRetryTime = 1;
  AddLinkConn(&rasNextLink, conn, 1);
  AddLinkConn(&rasNextLink, MakeConn(1002), 2);
  MakeQueuedMsg(conn, RAS_MSG_COLLREQ, 1);
  rasPfds[0].events = POLLIN;
  rasConnResume(conn);
  EXPECT_EQ(0, conn->startRetryTime);
  EXPECT_EQ(nullptr, rasNextLink.conns->next);
  EXPECT_TRUE(rasPfds[0].events & POLLOUT);
  ASSERT_FALSE(g_events.empty());
  EXPECT_EQ("PEER_CONNECTED", g_events.back().type);
}

TEST_F(RasNetMicrotest, LinkFallbackOpensCalculatedPeer) {
  SetPeers({1001, 1002, 1003});
  rasConnection* failed = MakeConn(1001);
  failed->experiencingDelays = true;
  AddLinkConn(&rasNextLink, failed, 0);
  g_calculatedPeers.push_back(1);
  ASSERT_EQ(ncclSuccess, rasLinkAddFallback(&rasNextLink, failed));
  ASSERT_NE(nullptr, rasNextLink.conns->next);
  EXPECT_EQ(1, rasNextLink.conns->next->peerIdx);
  EXPECT_NE(nullptr, rasNextLink.conns->next->conn);
}

TEST_F(RasNetMicrotest, LinkFallbackPrefersEarlierHealthyExternalConnection) {
  SetPeers({1001, 1002, 1003});
  rasConnection* failed = MakeConn(1001);
  rasConnection* external = MakeConn(1002);
  AddLinkConn(&rasNextLink, failed, 0);
  rasLinkConn* externalEntry = AddLinkConn(&rasNextLink, external, 1, true);
  g_calculatedPeers.push_back(2);
  ASSERT_EQ(ncclSuccess, rasLinkAddFallback(&rasNextLink, failed));
  EXPECT_FALSE(externalEntry->external);
  EXPECT_EQ(nullptr, externalEntry->next);
}

TEST_F(RasNetMicrotest, LinkFallbackInsertsAheadOfLaterExternalConnection) {
  SetPeers({1001, 1002, 1003});
  rasConnection* failed = MakeConn(1001);
  rasConnection* external = MakeConn(1003);
  AddLinkConn(&rasNextLink, failed, 0);
  AddLinkConn(&rasNextLink, external, 2, true);
  g_calculatedPeers.push_back(1);
  ASSERT_EQ(ncclSuccess, rasLinkAddFallback(&rasNextLink, failed));
  EXPECT_EQ((std::vector<int>{0, 1, 2}), LinkPeerIndices(rasNextLink));
  EXPECT_TRUE(rasNextLink.conns->next->next->external);
}

TEST_F(RasNetMicrotest, LinkFallbackSkipsDelayedCandidateAndContinues) {
  SetPeers({1001, 1002, 1003});
  rasConnection* failed = MakeConn(1001);
  rasConnection* delayed = MakeConn(1002);
  delayed->experiencingDelays = true;
  AddLinkConn(&rasNextLink, failed, 0);
  g_calculatedPeers.push_back(1);
  g_calculatedPeers.push_back(2);
  ASSERT_EQ(ncclSuccess, rasLinkAddFallback(&rasNextLink, failed));
  EXPECT_EQ((std::vector<int>{0, 1, 2}), LinkPeerIndices(rasNextLink));
  EXPECT_EQ(delayed, rasNextLink.conns->next->conn);
  EXPECT_FALSE(rasNextLink.conns->next->next->conn->experiencingDelays);
}

TEST_F(RasNetMicrotest, LinkFallbackStopsForHealthyOrExternalAlternatives) {
  rasConnection* failed = MakeConn(1001);
  rasConnection* healthy = MakeConn(1002);
  AddLinkConn(&rasNextLink, failed, 0);
  AddLinkConn(&rasNextLink, healthy, 1);
  EXPECT_EQ(ncclSuccess, rasLinkAddFallback(&rasNextLink, failed));
  EXPECT_TRUE(g_calculatedPeers.empty());

  FreeLink(&rasNextLink);
  AddLinkConn(&rasNextLink, failed, 0, true);
  EXPECT_EQ(ncclSuccess, rasLinkAddFallback(&rasNextLink, failed));
}

TEST_F(RasNetMicrotest, LinkFallbackIgnoresUninitializedPeerAndHandlesExhaustion) {
  rasConnection* failed = MakeConn();
  AddLinkConn(&rasNextLink, failed, -1);
  EXPECT_EQ(ncclSuccess, rasLinkAddFallback(&rasNextLink, failed));
  rasNextLink.conns->peerIdx = 0;
  EXPECT_EQ(ncclSuccess, rasLinkAddFallback(&rasNextLink, failed));
}

TEST_F(RasNetMicrotest, TerminateReleasesLinksConnectionsSocketsAndMessages) {
  rasConnection* conn = MakeConn();
  MakeQueuedMsg(conn, RAS_MSG_COLLREQ, 1);
  MakeSock(conn, RAS_SOCK_READY, 0);
  AddLinkConn(&rasNextLink, conn, 0);
  AddLinkConn(&rasPrevLink, conn, 0);
  MakeSock(nullptr, RAS_SOCK_READY, 1);
  rasNextLink.lastUpdatePeersTime = 4;
  rasPrevLink.lastUpdatePeersTime = 5;
  rasNetTerminate();
  EXPECT_EQ(nullptr, rasNextLink.conns);
  EXPECT_EQ(nullptr, rasPrevLink.conns);
  EXPECT_EQ(0, rasNextLink.lastUpdatePeersTime);
  EXPECT_EQ(0, rasPrevLink.lastUpdatePeersTime);
  EXPECT_EQ(nullptr, rasConnsHead);
  EXPECT_EQ(nullptr, rasSocketsHead);
}
