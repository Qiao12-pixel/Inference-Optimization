# mini-llama.cpp

一个用 C++17 实现的轻量级 LLM 推理引擎，用于理解和验证 Qwen2 / LLaMA-style
Decoder-only Transformer 的推理、量化、KV Cache 和 CUDA 性能优化路径。

项目支持真实 `Qwen2-0.5B-Instruct` GGUF 模型，包含 CPU 多线程、NVIDIA CUDA
后端，以及可复现的正确性和性能测试。

## 核心能力

- Qwen2/LLaMA-style Transformer：RMSNorm、RoPE、GQA causal attention、SwiGLU、残差连接与 KV Cache。
- 模型与 Tokenizer：GGUF v3、GGUF BPE tokenizer、chat template，以及 JSON + BIN 教学模型。
- 量化：F32、Q8_0、Q4_0、Q4_1 Linear 权重路径。
- 推理模式：单次生成、交互式聊天、模型检查、GGUF 检查和 benchmark CLI。
- 后端：CPU SIMD / 多线程，以及 CUDA F32/Q8_0/Q4_0/Q4_1 Linear、GPU KV Cache 和 GPU Attention。
- 验证：自研 C++ 测试、Python/NumPy reference forward、Golden Logits 与 CLI 回归测试。

## 实测性能

测试模型为 `Qwen2-0.5B-Instruct-Q8_0.gguf`，固定 seed、三次运行取中位数。

| 场景 | 环境 | 结果 |
|---|---|---:|
| CPU Decode，10 线程 | Apple M5 / 16 GB | 22.36 tok/s，较单线程 1.90× |
| CUDA Decode，Q8 warp + memory pool + Add/RMSNorm + 通用 Linear/ArgMax dispatcher | RTX 4090 / 24 GB | 357.69 tok/s，较 63.45 tok/s 基线 5.64× |
| CUDA Decode D2H，64 token | RTX 4090 | 38.90 MB → 256 B，降低 151,936× |
| CUDA Prefill，1024 token | RTX 4090 | 19.07 s → 1.32 s，14.42× |
| CUDA Prefill 吞吐，1024 token | RTX 4090 | 53.71 → 774.30 tok/s，+1,341.6% |
| 可选 Q8 F32-cache + GQA shared-KV Prefill，1024 token | RTX 4090 | 1.33 s → 110 ms，12.02×；额外 1,365 MB VRAM |
| Q8 FP16 Tensor Core Prefill，1024 token | RTX 4090 | 318 ms；cache 减半至 682.5 MB，但较 F32-cache 慢 25.7% |

Prefill 优化将顺序 token-by-token 前向改为 sequence-batched QKV/FFN、批量
RMSNorm/RoPE、连续 KV 写入和 tiled online-softmax causal attention。长 Prompt 下 CUDA
Linear/Activation/Attention 调用数固定为 `169 / 217 / 24`，不再随 token 数线性增长。

完整测试条件、原始数据和优化过程见：

- [项目实测与简历材料](docs/project-resume.md)
- [CUDA Decode / Prefill 优化记录](docs/cuda-decode-optimization.md)
- [RTX 4090 顺序 Prefill 基准](docs/benchmarks/rtx4090-prefill-profile.md)
- [RTX 4090 Batched Prefill 基准](docs/benchmarks/rtx4090-prefill-batched-profile.md)
- [RTX 4090 Tiled Attention Prefill 基准](docs/benchmarks/rtx4090-prefill-tiled-attention-profile.md)
- [RTX 4090 Last-token-only Prefill 基准](docs/benchmarks/rtx4090-prefill-last-logit-profile.md)
- [RTX 4090 Q8 F32-cache GEMM Prefill 基准](docs/benchmarks/rtx4090-prefill-q8-f32-gemm-profile.md)
- [RTX 4090 Q8 F32-cache Handle-cache 基准](docs/benchmarks/rtx4090-prefill-q8-f32-handle-cache-profile.md)
- [RTX 4090 GQA Shared-KV Prefill 基准](docs/benchmarks/rtx4090-prefill-gqa-shared-kv-profile.md)
- [RTX 4090 Q8 FP16 Tensor Core Prefill 基准](docs/benchmarks/rtx4090-prefill-q8-f16-tensorcore-profile.md)

## 推理流程

```text
GGUF / JSON+BIN
       │
       ▼
Model Loader + Tokenizer + Chat Template
       │
       ▼
Prefill / Decode
       │
       ├── CPU: SIMD + persistent thread pool
       └── CUDA: resident weights + KV Cache + attention + GPU ArgMax
       │
       ▼
Sampler / generated text
```

## 构建

依赖：CMake ≥ 3.17、支持 C++17 的编译器。运行 Python reference/Golden 测试时需要 Python 3 和 NumPy。

### CPU

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j4
ctest --test-dir build-release --output-on-failure
```

### CUDA（RTX 4090）

需要 NVIDIA CUDA Toolkit；`89` 是 RTX 4090 的 Ada compute capability。

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DMINI_LLAMA_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build-cuda -j4
ctest --test-dir build-cuda --output-on-failure
```

## 运行

### 单次生成

```bash
./build-release/mini-llama generate \
  --model models/tiny/model.bin \
  --config models/tiny/model.json \
  --tokenizer models/tiny/vocab.json \
  --prompt hello --n-predict 16
```

### Qwen2 交互聊天

```bash
./build-release/mini-llama run models/chat -n 64
```

### RTX 4090 benchmark

```bash
./build-cuda/mini-llama bench \
  models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf \
  --prompt hello --n-predict 64 --seed 42 \
  --backend cuda --device 0
```

### 精确长度的 Prefill Profile

`--prompt-tokens` 使用重复 BOS token，避免不同文本分词带来的 token 数差异。

```bash
python3 scripts/run_prefill_profile.py \
  --binary build-cuda/mini-llama \
  --backend cuda \
  --out docs/benchmarks/rtx4090-prefill-profile.md
```

实验性 Q8 Prefill cuBLAS 路径会用额外显存缓存 F32 解量化权重，仅影响二维 Prefill，
不会替换 Q8 Decode kernel：

```bash
python3 scripts/run_prefill_profile.py \
  --binary build-cuda/mini-llama \
  --backend cuda --q8-prefill f32 \
  --out docs/benchmarks/rtx4090-prefill-q8-f32-gemm-profile.md
```

实验性 FP16 Tensor Core 路径将额外 Q8 cache 压缩至约 F32 路径的一半；需先验证
logits 和 token 一致性：

```bash
python3 scripts/run_prefill_profile.py \
  --binary build-cuda/mini-llama \
  --backend cuda --q8-prefill f16 \
  --out docs/benchmarks/rtx4090-prefill-q8-f16-tensorcore-profile.md
```

cuBLAS handle 已按 host thread / CUDA device 复用；重新评估 F32-cache Prefill：

```bash
python3 scripts/run_prefill_profile.py \
  --binary build-cuda/mini-llama \
  --backend cuda --q8-prefill f32 \
  --out docs/benchmarks/rtx4090-prefill-q8-f32-handle-cache-profile.md
```

可以让 runtime 根据可用显存自动选择长 Prompt Prefill profile：

```bash
./build-cuda/mini-llama bench \
  models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf \
  --prompt-tokens 1024 --n-predict 1 --seed 42 \
  --backend cuda --q8-prefill auto
```

`auto` 仅在 F32 cache、完整 KV Cache 和 256 MiB runtime 余量均可容纳时选择 F32-cache；
否则回退到默认 Q8 path。可显式使用 `--q8-prefill q8|f32|f16` 固定策略。

已在 RTX 4090 验证 `auto → f32`：1024-token 请求的额外 F32 cache 为 1,365 MB，
总 GPU 权重内存为 2,766.48 MB；单次 smoke Prefill 为 204.53 ms。正式性能报告仍使用
三次中位数 `207.06 ms / 4,945.43 tok/s`。

### Nsight Systems hotspot profile

在进一步优化 Q8 Linear 或 tiled attention 前，先采集独立的 Prefill 与 Decode
profile。脚本输出 `.nsys-rep`、CUDA kernel/API/memory 汇总和运行 manifest。

```bash
python3 scripts/run_nsys_profile.py \
  --binary build-cuda/mini-llama \
  --out-dir docs/profiles/rtx4090
```

对长 Prompt 高吞吐 F32-cache profile，显式指定策略，避免将默认 Q8 trace 用于错误的
热点判断：

```bash
python3 scripts/run_nsys_profile.py \
  --binary build-cuda/mini-llama \
  --prefill-tokens 1024 --q8-prefill f32 \
  --out-dir docs/profiles/rtx4090-f32-pool
```

当前 F32-cache Profile 将 Prefill 主要热点定位为 GQA attention；实验性 shared-KV
attention 使用 `(token, KV head)` block 复用 K/V tile。验证它时使用独立报告：

```bash
python3 scripts/run_prefill_profile.py \
  --binary build-cuda/mini-llama \
  --backend cuda --q8-prefill f32 \
  --out docs/benchmarks/rtx4090-prefill-gqa-shared-kv-profile.md
```

用 Nsight Systems GUI 打开 `prefill.nsys-rep` 或 `decode.nsys-rep`，也可直接阅读
对应的 `*-summary.txt`。

### Q8 Decode kernel profile

对单 token Q8 Linear 的 warp-per-output kernel，使用 Nsight Compute 采集
occupancy、memory throughput 和 warp stall：

```bash
python3 scripts/run_ncu_q80_profile.py \
  --binary build-cuda/mini-llama \
  --out docs/profiles/rtx4090/q80-decode-warp
```

默认只采样一个匹配 kernel；使用 `--launch-skip N` 可选取 Decode 内不同位置的
Q8 Linear 进行分析，避免对全部 launch 做昂贵的 Nsight Compute replay。

## 测试与正确性

- Apple M5 Release 环境：自研测试运行器 `214/214` 通过，CTest `23/23` 通过；RTX 4090 CUDA 环境：量化 Linear+ArgMax 融合构建完整 CTest `23/23` 通过（16.96 s）。
- C++ 推理与 Python/NumPy reference forward、Golden Logits 对齐。
- CUDA 覆盖测试包括量化 Linear、device-resident forward、KV Cache、Attention 与 CUDA runtime。

## 仓库结构

```text
include/mini_llama/   Public interfaces
src/                  Runtime, Transformer forward, CPU/CUDA kernels
tests/                C++ unit and CUDA path tests
scripts/              Reference forward, model tools, regression, profiling
models/tiny/          Small deterministic test model
models/chat/          Qwen2 GGUF model and tokenizer assets
docs/                 Design notes, benchmarks, resume-ready results
```

## 当前边界

- CUDA Top-K / Top-P / temperature sampling 仍会回退 CPU；默认 greedy sampling 已使用 GPU ArgMax。
- Q4_0 已验证速度与内存收益，但当前 logits 误差较大，不能视为无损量化。
- Batched Prefill 使用教学性质的 causal attention kernel，后续可继续探索 FlashAttention-style tiling 与 Tensor Core 量化 GEMM。
- RTX 4090 Nsight Systems 显示 Q8_0 Linear 是当前主热点；Decode 通过 warp-per-output kernel、stream-ordered memory pool、Add/RMSNorm 与通用 `CudaLinearArgMax` dispatcher 的 F32 融合路径优化至 357.69 tok/s。dispatcher 已覆盖 F32/Q8_0/Q4_0/Q4_1 的候选归约实现，量化算子和端到端 CUDA 回归均通过；量化 lm_head 性能仍需以使用量化 lm_head 的真实模型单独测量。Prefill 的下一方向仍是专用 Q8 tiled GEMM / Tensor Core 路径。
- CUDA runtime 使用 stream-ordered memory pool（`cudaMallocAsync/cudaFreeAsync`）复用短生命周期 activation buffer；需在目标 CUDA 环境完成 benchmark 回归验证。
- 尚未与同机 `llama.cpp` 做公平对照，因此不主张性能达到或超过 `llama.cpp`。

## License

[MIT](LICENSE)
