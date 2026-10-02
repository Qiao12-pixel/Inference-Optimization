// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/thread_pool.h"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace mini_llama {

namespace {

int g_thread_count = 0;

class PersistentThreadPool {
 public:
  void Run(int requested_threads, int n,
           const std::function<void(int begin, int end)>& fn) {
    EnsureWorkerCount(requested_threads);

    std::unique_lock<std::mutex> lock(mutex_);
    const int task_count = std::min(requested_threads, n);
    const int chunk = n / task_count;
    const int remainder = n % task_count;
    starts_.assign(static_cast<size_t>(task_count), 0);
    ends_.assign(static_cast<size_t>(task_count), 0);
    int start = 0;
    for (int task = 0; task < task_count; ++task) {
      const int count = chunk + (task < remainder ? 1 : 0);
      starts_[static_cast<size_t>(task)] = start;
      ends_[static_cast<size_t>(task)] = start + count;
      start += count;
    }

    fn_ = &fn;
    first_exception_ = nullptr;
    active_tasks_ = task_count;
    remaining_tasks_ = task_count;
    ++generation_;
    work_cv_.notify_all();
    done_cv_.wait(lock, [&] { return remaining_tasks_ == 0; });
    fn_ = nullptr;
    if (first_exception_) {
      std::rethrow_exception(first_exception_);
    }
  }

 private:
  void EnsureWorkerCount(int requested_threads) {
    if (requested_threads <= 0) {
      throw std::runtime_error(
          "PersistentThreadPool: thread count must be positive");
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (workers_.size() == static_cast<size_t>(requested_threads)) {
        return;
      }
    }

    StopWorkers();

    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = false;
    workers_.reserve(static_cast<size_t>(requested_threads));
    for (int worker = 0; worker < requested_threads; ++worker) {
      workers_.emplace_back([this, worker] { WorkerLoop(worker); });
    }
  }

  void StopWorkers() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (workers_.empty()) {
        return;
      }
      stopping_ = true;
      work_cv_.notify_all();
    }
    for (std::thread& worker : workers_) {
      worker.join();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    workers_.clear();
    starts_.clear();
    ends_.clear();
    generation_ = 0;
    active_tasks_ = 0;
    remaining_tasks_ = 0;
    fn_ = nullptr;
  }

  void WorkerLoop(int worker_index) {
    int observed_generation = 0;
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      work_cv_.wait(lock, [&] {
        return stopping_ || generation_ != observed_generation;
      });
      if (stopping_) {
        return;
      }
      observed_generation = generation_;
      if (worker_index >= active_tasks_) {
        continue;
      }

      const int begin = starts_[static_cast<size_t>(worker_index)];
      const int end = ends_[static_cast<size_t>(worker_index)];
      const std::function<void(int, int)>* fn = fn_;
      lock.unlock();

      std::exception_ptr exception;
      try {
        (*fn)(begin, end);
      } catch (...) {
        exception = std::current_exception();
      }

      lock.lock();
      if (exception && !first_exception_) {
        first_exception_ = exception;
      }
      --remaining_tasks_;
      if (remaining_tasks_ == 0) {
        done_cv_.notify_one();
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  std::vector<std::thread> workers_;
  std::vector<int> starts_;
  std::vector<int> ends_;
  const std::function<void(int, int)>* fn_ = nullptr;
  std::exception_ptr first_exception_;
  int generation_ = 0;
  int active_tasks_ = 0;
  int remaining_tasks_ = 0;
  bool stopping_ = false;
};

PersistentThreadPool& GetPersistentThreadPool() {
  // Intentionally process-lifetime: worker shutdown during static destruction
  // is fragile, while the OS safely releases resources at program exit.
  static PersistentThreadPool* pool = new PersistentThreadPool();
  return *pool;
}

}  // namespace

int GetThreadCount() {
  if (g_thread_count > 0) {
    return g_thread_count;
  }
  const unsigned int hw = std::thread::hardware_concurrency();
  return hw > 0 ? static_cast<int>(hw) : 4;
}

void SetThreadCount(int n) { g_thread_count = n > 0 ? n : 0; }

void ParallelFor(int n, const std::function<void(int begin, int end)>& fn) {
  if (n <= 0) {
    return;
  }

  const int n_threads = GetThreadCount();
  constexpr int kMinChunk = 16;
  if (n_threads <= 1 || n < n_threads * kMinChunk) {
    fn(0, n);
    return;
  }
  GetPersistentThreadPool().Run(n_threads, n, fn);
}

}  // namespace mini_llama
