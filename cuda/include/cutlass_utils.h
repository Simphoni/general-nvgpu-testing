#pragma once
#include <cstdio>

#include "cutlass/cutlass.h"

#define cutlassSafeCall(err) __cutlassSafeCall(err, __FILE__, __LINE__)

inline void __cutlassSafeCall(cutlass::Status err, const char *file,
                              const int line) {
  if (cutlass::Status::kSuccess != err) {
    fprintf(stderr, "cutlassSafeCall() failed at %s:%i : %s\n", file, line,
            cutlassGetStatusString(err));
    exit(-1);
  }
}