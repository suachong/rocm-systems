// Experimental 4-node, 8-local-rank hierarchical RS peer-read arguments.
// Passed by value: no asynchronous dependency on a host pointer table.
#ifndef RCCL_CE_RS_PEER_H_
#define RCCL_CE_RS_PEER_H_
#include <stddef.h>
struct ncclCeRsPeerArgs {
  const void* peers[8];
  size_t targetOffsets[4];
  size_t count;
  size_t stride;
};
#endif
