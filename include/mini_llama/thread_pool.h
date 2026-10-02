// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_THREAD_POOL_H_
#define INCLUDE_MINI_LLAMA_THREAD_POOL_H_

#include <functional>
#include <thread>
#include <vector>

namespace mini_llama {

// Get the configured number of threads for parallel operations.
// Defaults to std::thread::hardware_concurrency().
int GetThreadCount();

// Set the number of threads for parallel operations.
// Pass 0 to Reset to hardware_concurrency().
void SetThreadCount(int n);

// Persistent parallel for: split [0, n) into roughly equal chunks. Worker
// threads are reused between calls to avoid per-operator thread creation.
void ParallelFor(int n, const std::function<void(int begin, int end)>& fn);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_THREAD_POOL_H_
