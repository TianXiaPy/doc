# cuBLASMp 新手入门与最小 GEMM 示例

[cuBLASMp](https://docs.nvidia.com/cuda/cublasmp/index.html) 是面向**多进程 / 多 GPU** 的稠密线性代数库（类似 PBLAS），矩阵采用 **2D block-cyclic** 分布，数据在 **GPU 显存** 上。

## 重要说明

- **官方支持操作系统为 Linux**（x86_64 / arm64-sbsa）。若在 Windows 上自学，请使用 **WSL2（GPU）**、虚拟机或 Linux 服务器。
- 需要 **MPI**、**NCCL**，以及文档中要求的 **CUDA / NVSHMEM** 等版本，见 [Getting Started](https://docs.nvidia.com/cuda/cublasmp/getting_started/index.html)。
- 本仓库的 `gemm_minimal.cu` 固定使用 **2 个 MPI 进程**、进程网格 **2×1**，单机双卡或单卡可跑（多进程可共享 GPU，仅适合练手）。

## 推荐学习路径（快速入门）

1. **先建立概念**：全局矩阵不存在于单一 GPU 上，每个 rank 只持有本地分块；`cublasMpNumroc` 计算本地行/列块数，`lld` 为本地列主矩阵的 leading dimension。
2. **读官方文档**：[How to Use cuBLASMp](https://docs.nvidia.com/cuda/cublasmp/usage/index.html) → [NCCL Initialization](https://docs.nvidia.com/cuda/cublasmp/usage/initialization/nccl.html)（MPI + `ncclCommInitRank` + `cublasMpGridCreate`）。
3. **对照完整示例**：官方维护的 [CUDALibrarySamples/cuBLASMp](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuBLASMp)（含 `gemm.cu`、CMake、矩阵生成与命令行参数）。
4. **获取库**：从 [NVIDIA 下载页](https://developer.nvidia.com/cublasmp-downloads)、PyPI/Conda 或 **HPC SDK** 安装 cuBLASMp，并保证 **NCCL** 与 MPI 可用。

## 编译本示例（Linux）

```bash
export CUBLASMP_HOME=/your/cublasmp   # include/ 与 lib/libcublasmp.so
export NCCL_HOME=/your/nccl           # 含 nccl.h 与 libnccl.so
# 加载 MPI 模块或 source HPC-X 环境脚本

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCUBLASMP_ROOT="${CUBLASMP_HOME}" \
  -DNCCL_ROOT="${NCCL_HOME}" \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
```

## 运行

```bash
mpirun -n 2 ./build/gemm_minimal
```

若进程数不是 2，程序会提示应使用 `mpirun -n 2`（与代码内 `nprow=2,npcol=1` 一致）。

## 文件说明

| 文件 | 作用 |
|------|------|
| `gemm_minimal.cu` | 最小 `cublasMpGemm`：初始化 MPI/NCCL、建 grid/descriptor、分配 workspace、调用 GEMM |
| `CMakeLists.txt` | 链接 `cublasmp`、`nccl`、`MPI`、`cudart` 的示例配置 |

更复杂的张量并行、AllGather+GEMM 等见官方文档 [Tensor Parallelism](https://docs.nvidia.com/cuda/cublasmp/usage/tp.html)。
