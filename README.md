# SUMMA AB 分布式GEMM实现 (NCCL + cuBLAS + MPI)

## 概述

这是一个基于 **SUMMA AB算法** 的分布式矩阵乘法(GEMM)实现，使用纯 NCCL + cuBLAS + MPI，**不依赖cublasMp**。

- **算法**: SUMMA AB (Scalable Universal Matrix Multiplication Algorithm, Broadcast-Broadcast Variant)
- **目标平台**: NVIDIA RTX 4090 (sm_89)
- **通信库**: NCCL (GPU间高速通信)
- **计算库**: cuBLAS (本地GEMM)
- **进程管理**: MPI (多进程/多节点)

## SUMMA AB算法原理

### 问题定义
计算: `C = alpha * A * B + beta * C`

- A: M × K 矩阵
- B: K × N 矩阵  
- C: M × N 矩阵

### 数据分布 (2D Block-Cyclic)

```
进程网格: nprow × npcol
┌─────────────┬─────────────┐
│   (0,0)     │   (0,1)     │
│   Rank 0    │   Rank 1    │
├─────────────┼─────────────┤
│   (1,0)     │   (1,1)     │
│   Rank 2    │   Rank 3    │
└─────────────┴─────────────┘
```

每个进程仅存储矩阵的一个2D子块:
- A: 本地大小 = mloc_A × nloc_A
- B: 本地大小 = mloc_B × nloc_B  
- C: 本地大小 = mloc_C × nloc_C

### 算法步骤

```
for k = 0 to K_steps-1:
    // 1. 广播A的第k列块到同行所有进程
    A_owner = k % nprow
    Bcast(A_panel, A_owner, row_comm)
    
    // 2. 广播B的第k行块到同列所有进程  
    B_owner = k % npcol
    Bcast(B_panel, B_owner, col_comm)
    
    // 3. 本地GEMM计算
    C += A_panel × B_panel
```

### 通信模式

SUMMA AB的关键是**双向广播**:
1. **水平广播**: A的数据沿行广播 (使用 nccl_row_comm)
2. **垂直广播**: B的数据沿列广播 (使用 nccl_col_comm)

## 文件结构

```
summa_gemm_v2.cu  - 主要实现文件
summa_gemm.cu     - 早期版本 (参考)
Makefile          - 编译脚本
cublasmp_matmul_mpi.cu  - cublasMp参考代码
SUMMA_README.md   - 本文档
```

## 编译

### Linux (推荐)

```bash
# 使用mpicxx (自动包含MPI头文件和库)
mpicxx -O3 -o summa_gemm summa_gemm_v2.cu \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -lcudart -lcublas -lnccl \
    -gencode arch=compute_89,code=sm_89

# 或使用Makefile
make linux
```

### Windows

```powershell
# 使用nvcc直接编译
nvcc -O3 -o summa_gemm.exe summa_gemm_v2.cu ^
    -I"C:\Program Files\Microsoft MPI\Include" ^
    -L"C:\Program Files\Microsoft MPI\Lib\x64" ^
    -lmsmpi -lcudart -lcublas -lnccl ^
    -arch=sm_89

# 或使用Makefile
make
```

### 编译参数说明

- `-arch=sm_89`: 针对RTX 4090 (Ada架构)
- `-O3`: 最高优化级别
- `-lcudart`: CUDA运行时
- `-lcublas`: cuBLAS库
- `-lnccl`: NCCL通信库
- `-lmpi`: MPI库

## 运行

### 基本用法

```bash
mpirun -np 4 ./summa_gemm <nprow> <npcol> <M> <N> <K> <mb> <nb> <kb>
```

参数:
- `nprow`: 进程网格行数
- `npcol`: 进程网格列数
- `M,N,K`: 矩阵维度 (A:M×K, B:K×N, C:M×N)
- `mb,nb,kb`: 分块大小

### 示例

```bash
# 4个GPU，2×2网格，4096×4096矩阵
mpirun -np 4 ./summa_gemm 2 2 4096 4096 4096 1024 1024 1024

# 8个GPU，2×4网格，8192×8192矩阵
mpirun -np 8 ./summa_gemm 2 4 8192 8192 8192 1024 1024 1024

# 使用Makefile运行
make run-4gpu
```

### 输出示例

```
========================================
SUMMA AB Distributed GEMM
========================================
Process Grid: 2 x 2 = 4 GPUs
Matrix: C(4096 x 4096) = A(4096 x 4096) * B(4096 x 4096)
Block Sizes: mb=1024, nb=1024, kb=1024
========================================
Local Matrix Sizes:
  A: 2048 x 4096
  B: 4096 x 2048
  C: 2048 x 2048
========================================
Running SUMMA GEMM...

========================================
Results:
  Time:      0.2345 seconds
  GFLOPS:    585.23
  GFLOPS/GPU: 146.31
  Bandwidth: 128.45 GB/s (estimated)
========================================
```

## 性能调优

### 1. 分块大小 (mb, nb, kb)

分块大小影响:
- **通信量**: 块越小，通信次数越多
- **计算效率**: 块太小会导致cuBLAS效率下降
- **内存占用**: 临时缓冲区大小

**推荐设置**:
- 对于4090: `mb=nb=kb=1024` 或 `2048`
- 对于大矩阵(>8192): `mb=nb=kb=2048`

### 2. 进程网格形状

```
# 4 GPU的两种配置
2×2网格: 通信最均衡
1×4网格: 适合M小N大的矩阵
4×1网格: 适合M大N小的矩阵
```

### 3. 通信优化

- 使用 `cudaStreamSynchronize` 控制同步点
- NCCL广播可以与计算重叠 (需要额外实现)
- 考虑使用CUDA Graph捕获重复模式

## 算法对比

| 特性 | SUMMA AB (本实现) | cublasMp | 纯MPI (CPU) |
|------|------------------|----------|-------------|
| 通信库 | NCCL (GPU-GPU) | NCCL+NVSHMEM | MPI (CPU内存) |
| 计算 | cuBLAS | 内部优化 | MKL/OpenBLAS |
| 适用场景 | 多GPU单机/多机 | NVIDIA官方 | 纯CPU集群 |
| 可控性 | 完全可控 | 黑盒 | 中等 |
| 代码复杂度 | 中等 | 低 | 低 |

## 扩展与优化建议

### 1. 计算与通信重叠

```cuda
// 当前实现: 阻塞式
for k in steps:
    ncclBcast(B_panel)      // 通信
    cudaStreamSynchronize()  // 等待
    cublasSgemm(...)         // 计算

// 优化: 双缓冲重叠
for k in steps:
    ncclBcast(B_panel_next, next_stream)  // 预取下一轮
    cublasSgemm(A_panel, B_panel_curr, ...) // 当前轮计算
    swap_buffers()
```

### 2. 支持其他数据类型

修改 `float` → `half` (FP16) 或 `__nv_bfloat16` (BF16):
```cuda
// 使用cublasGemmEx支持混合精度
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    A, CUDA_R_16F, lda,
    B, CUDA_R_16F, ldb,
    &beta,
    C, CUDA_R_16F, ldc,
    CUBLAS_COMPUTE_32F,  // 计算用FP32
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

### 3. 支持批量GEMM

对于多个小矩阵相乘，可以批量处理提高效率。

## 调试技巧

### 1. 检查NCCL版本
```bash
nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 4
```

### 2. 验证数据分布
```bash
# 打印每个进程的本地矩阵大小
mpirun -np 4 ./summa_gemm 2 2 4096 4096 4096 1024 1024 1024 | grep "Rank"
```

### 3. 使用Nsight分析
```bash
nsys profile -o summa_report mpirun -np 4 ./summa_gemm 2 2 4096 4096 4096 1024 1024 1024
```

## 已知限制

1. **内存需求**: 需要额外的临时缓冲区存储A_panel和B_panel
2. **同步开销**: 每次迭代需要等待NCCL广播完成
3. **K维度**: 要求K能被kb整除 (当前实现)

## 参考资源

- [SUMMA Paper](https://www.netlib.org/lapack/lawnspdf/lawn96.pdf): 原始算法论文
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [ScaLAPACK User Guide](https://netlib.org/scalapack/slug/): 2D块循环分布参考

## 联系与支持

如有问题，请检查:
1. NCCL是否正确安装: `nccl.h` 和 `libnccl.so`
2. MPI是否支持GPU: 建议使用OpenMPI 4.0+或MPICH 3.4+
3. CUDA版本兼容性: 建议使用CUDA 11.8+ 或 12.x
