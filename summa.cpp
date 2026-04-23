// =============================================================================
// SUMMA_NN_AB_GEMM - 分布式矩阵乘法实现
// =============================================================================
// 本文件实现了基于 SUMMA AB 算法的分布式 GEMM (C = alpha * A * B + beta * C)
// 使用 MPI + NCCL + cuBLAS 技术栈，适用于多 GPU 环境
// =============================================================================

// 平台检测：NCCL 仅支持 Linux/WSL2，不支持原生 Windows
#if defined(_WIN32) && !defined(__linux__)
#error "NCCL is supported on Linux/WSL2 rather than native Windows. Build this demo on Ubuntu/WSL2 on the 4090 machine."
#endif

// SUMMA_NN_AB_GEMM - 算法名称说明：
// C = alpha * A * B + beta * C
//
// This implementation uses:
// - MPI   : rank management and process-grid setup
// - NCCL  : row-wise and column-wise panel broadcast
// - cuBLAS: local SGEMM on each GPU
//
// Distribution:
// - M dimension is block-cyclic over process rows with block size mb
// - N dimension is block-cyclic over process cols with block size nb
// - K dimension is panelized with size kb
// - A panel k is owned by process-column (k_step % npcol)
// - B panel k is owned by process-row    (k_step % nprow)
//
// Recommended build on a 4x4090 Linux/WSL2 box:
//   nvcc -O3 -std=c++17 -ccbin mpicxx \
//       -gencode arch=compute_89,code=sm_89 \
//       -o summa_gemm summa_gemm.cu \
//       -lcublas -lnccl
//
// Example run on a 2x2 GPU grid:
//   mpirun -np 4 ./summa_gemm 2 2 8192 8192 8192 1024 1024 1024 1.0 0.0 0

// =============================================================================
// 头文件包含
// =============================================================================
#include <mpi.h>           // MPI: 进程管理与通信
#include <nccl.h>          // NCCL: GPU 间高速通信
#include <cuda_runtime.h>  // CUDA: GPU 计算基础
#include <cublas_v2.h>     // cuBLAS: GPU 上的 BLAS 库

// 标准库头文件
#include <algorithm>       // 标准算法
#include <cmath>           // 数学函数
#include <cstdint>         // 整数类型定义
#include <cstdio>          // 输入输出
#include <cstdlib>         // 标准库函数
#include <cstring>         // 字符串操作
#include <limits>          // 数值限制
#include <vector>          // 动态数组

static int g_world_rank = -1;  // 全局 MPI 进程号，用于错误报告中标识进程

// =============================================================================
// 错误检查宏定义 - 自动检查 API 调用返回值并在出错时打印信息并终止程序
// =============================================================================

// MPI 错误检查宏：检查 MPI 函数返回值，出错时打印错误信息并终止
#define MPI_CHECK(cmd) do { \
    const int status_ = (cmd); \
    if (status_ != MPI_SUCCESS) { \
        fprintf(stderr, "[rank %d] MPI error at %s:%d, code=%d\n", \
                g_world_rank, __FILE__, __LINE__, status_); \
        std::abort(); \
    } \
} while (0)

// CUDA 错误检查宏：检查 CUDA 函数返回值，出错时打印错误描述并终止
#define CUDA_CHECK(cmd) do { \
    const cudaError_t status_ = (cmd); \
    if (status_ != cudaSuccess) { \
        fprintf(stderr, "[rank %d] CUDA error at %s:%d: %s\n", \
                g_world_rank, __FILE__, __LINE__, cudaGetErrorString(status_)); \
        std::abort(); \
    } \
} while (0)

// NCCL 错误检查宏：检查 NCCL 函数返回值，出错时打印错误描述并终止
#define NCCL_CHECK(cmd) do { \
    const ncclResult_t status_ = (cmd); \
    if (status_ != ncclSuccess) { \
        fprintf(stderr, "[rank %d] NCCL error at %s:%d: %s\n", \
                g_world_rank, __FILE__, __LINE__, ncclGetErrorString(status_)); \
        std::abort(); \
    } \
} while (0)

// cuBLAS 错误检查宏：检查 cuBLAS 函数返回值，出错时打印错误代码并终止
#define CUBLAS_CHECK(cmd) do { \
    const cublasStatus_t status_ = (cmd); \
    if (status_ != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[rank %d] cuBLAS error at %s:%d: %d\n", \
                g_world_rank, __FILE__, __LINE__, static_cast<int>(status_)); \
        std::abort(); \
    } \
} while (0)

// =============================================================================
// 进程网格结构体 - 管理 MPI 进程的拓扑布局和通信器
// =============================================================================
// 在 SUMMA 算法中，进程被组织成 2D 网格 (nprow x npcol)
// 每个进程根据其在网格中的位置参与不同的通信组
// =============================================================================
struct ProcGrid {
    // MPI 全局信息
    int world_rank = 0;     // 当前进程在 MPI_COMM_WORLD 中的 rank
    int world_size = 0;     // MPI 总进程数

    // 2D 网格维度
    int nprow = 1;          // 网格行数 (number of process rows)
    int npcol = 1;          // 网格列数 (number of process columns)

    // 当前进程在网格中的坐标
    int prow = 0;           // 当前进程所在行索引 (0 到 nprow-1)
    int pcol = 0;           // 当前进程所在列索引 (0 到 npcol-1)

    // 子通信器中的 rank
    int row_rank = 0;       // 在行通信器中的 rank
    int col_rank = 0;       // 在列通信器中的 rank

    // 本地节点信息 (同一物理节点的进程)
    int local_rank = 0;     // 在当前节点的 rank
    int local_size = 1;     // 当前节点的进程数

    // GPU 设备
    int device = 0;         // 当前进程绑定的 GPU 设备号

    // MPI 通信器 - 用于进程间协调
    MPI_Comm row_comm = MPI_COMM_NULL;    // 行通信器 (同一行的进程)
    MPI_Comm col_comm = MPI_COMM_NULL;    // 列通信器 (同一列的进程)
    MPI_Comm shared_comm = MPI_COMM_NULL; // 共享内存通信器 (同一节点)

    // NCCL 通信器 - 用于 GPU 间直接通信
    ncclComm_t row_nccl = nullptr;        // 行方向的 NCCL 通信器
    ncclComm_t col_nccl = nullptr;        // 列方向的 NCCL 通信器
};

// =============================================================================
// 本地矩阵结构体 - 表示每个进程持有的矩阵子块 (设备内存)
// =============================================================================
// 矩阵以列优先 (column-major) 格式存储，这是 Fortran/cuBLAS 的默认格式
// =============================================================================
struct LocalMatrix {
    float* d_ptr = nullptr;  // GPU 设备内存指针
    int64_t rows = 0;        // 本地行数 (本地高度)
    int64_t cols = 0;        // 本地列数 (本地宽度)
    int64_t ld = 1;          //  leading dimension (通常为 rows，即列间距)
};

// =============================================================================
// Panel 计划结构体 - 管理 K 维度的分块 (panel) 策略
// =============================================================================
// SUMMA 算法将 K 维度分割成多个 panel，每个 panel 大小为 kb
// 这些 panel 按轮转方式分配给不同进程列/行进行广播
// =============================================================================
struct PanelPlan {
    int64_t K = 0;           // K 维度总大小 (A 的列数 / B 的行数)
    int kb = 0;              // 每个 panel 的大小
    int steps = 0;           // panel 迭代次数 (ceil(K/kb))
    int64_t max_panel = 0;   // 最大 panel 大小 (处理 K 不能被 kb 整除的情况)

    // 本地存储的 panel 总量
    int64_t a_k_local = 0;   // 本进程 A 矩阵本地存储的 K 方向总大小
    int64_t b_k_local = 0;   // 本进程 B 矩阵本地存储的 K 方向总大小

    // 每个 panel 的详细信息
    std::vector<int64_t> starts;    // 每个 panel 在全局 K 维度的起始位置
    std::vector<int64_t> sizes;     // 每个 panel 的实际大小
    std::vector<int64_t> a_offsets; // 每个 panel 在本进程 A 矩阵中的列偏移 (-1 表示不拥有)
    std::vector<int64_t> b_offsets; // 每个 panel 在本进程 B 矩阵中的行偏移 (-1 表示不拥有)
};

// =============================================================================
// 辅助函数 - 数学工具和矩阵值生成
// =============================================================================

// 向上取整除法：计算 (a + b - 1) / b，即 ceil(a/b)
// 用于确定分块后的块数等场景
static inline int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

// 矩阵 A 元素值生成函数 - 基于行列索引生成确定性值
// 使用素数取模产生伪随机分布，用于测试验证
static inline float value_a(int64_t i, int64_t k) {
    return 1.0f
         + 0.0013f * static_cast<float>((i % 97) - 48)
         + 0.0007f * static_cast<float>((k % 89) - 44);
}

// 矩阵 B 元素值生成函数 - 基于行列索引生成确定性值
static inline float value_b(int64_t k, int64_t j) {
    return -0.5f
         + 0.0011f * static_cast<float>((k % 83) - 41)
         + 0.0009f * static_cast<float>((j % 79) - 39);
}

// 矩阵 C 元素值生成函数 (初始值) - 用于设置 beta*C 中的 C 初始值
static inline float value_c(int64_t i, int64_t j) {
    return 0.01f * static_cast<float>((i + 3 * j) % 23);
}

// =============================================================================
// 构建块循环分布的索引列表
// =============================================================================
// 参数:
//   n        - 全局维度大小 (行数或列数)
//   block    - 块大小 (mb 或 nb)
//   owner    - 当前进程在 owner 组中的索引
//   nowners  - owner 组中的总进程数 (nprow 或 npcol)
// 返回:
//   当前进程拥有的所有全局索引列表
//
// 例如: n=10, block=3, owner=0, nowners=2
//       块分布: [0,1,2] [3,4,5] [6,7,8] [9]
//       owner=0 得到块 0,2 -> 索引 [0,1,2,6,7,8]
//       owner=1 得到块 1,3 -> 索引 [3,4,5,9]
// =============================================================================
static std::vector<int64_t> build_block_cyclic_indices(
    int64_t n,        // 全局维度大小
    int64_t block,    // 块大小
    int owner,        // 当前进程索引
    int nowners) {    // owner 组大小
    std::vector<int64_t> indices;
    // 参数有效性检查
    if (n <= 0 || block <= 0 || nowners <= 0) {
        return indices;
    }

    // 计算总块数
    const int64_t nblocks = ceil_div(n, block);

    // 遍历当前进程拥有的所有块
    for (int64_t blk = owner; blk < nblocks; blk += nowners) {
        const int64_t begin = blk * block;                    // 块起始位置
        const int64_t width = std::min<int64_t>(block, n - begin);  // 块实际大小
        // 将块内所有索引加入结果
        for (int64_t x = 0; x < width; ++x) {
            indices.push_back(begin + x);
        }
    }
    return indices;
}

// =============================================================================
// 构建 Panel 计划 - 确定 K 维度的分块策略和每个进程的数据分布
// =============================================================================
// SUMMA AB 算法中，K 维度被划分为多个 panel，panel k 的分布规则：
//   - A 的第 k 个 panel 属于进程列 (k % npcol)
//   - B 的第 k 个 panel 属于进程行    (k % nprow)
//
// 这种轮转分布确保每个进程列/行都能参与广播
// =============================================================================
static PanelPlan build_panel_plan(
    int64_t K,        // K 维度总大小
    int kb,           // 每个 panel 的大小
    int prow,         // 当前进程行坐标
    int nprow,        // 进程网格行数
    int pcol,         // 当前进程列坐标
    int npcol) {      // 进程网格列数
    PanelPlan plan;
    plan.K = K;
    plan.kb = kb;
    plan.steps = static_cast<int>(ceil_div(K, static_cast<int64_t>(kb)));
    plan.starts.resize(plan.steps);
    plan.sizes.resize(plan.steps);
    plan.a_offsets.resize(plan.steps, -1);  // -1 表示当前进程不拥有此 panel
    plan.b_offsets.resize(plan.steps, -1);  // -1 表示当前进程不拥有此 panel

    int64_t a_offset = 0;  // A 矩阵本地 panel 累积偏移
    int64_t b_offset = 0;  // B 矩阵本地 panel 累积偏移

    // 遍历每个 panel，计算其位置和归属
    for (int step = 0; step < plan.steps; ++step) {
        const int64_t begin = static_cast<int64_t>(step) * kb;  // panel 全局起始
        const int64_t width = std::min<int64_t>(kb, K - begin); // panel 实际大小
        plan.starts[step] = begin;
        plan.sizes[step] = width;
        plan.max_panel = std::max(plan.max_panel, width);       // 记录最大 panel

        // panel step 属于进程列 (step % npcol)
        if ((step % npcol) == pcol) {
            plan.a_offsets[step] = a_offset;  // 记录 A 中此 panel 的偏移
            a_offset += width;                 // 累加本地 K 维度大小
        }
        // panel step 属于进程行 (step % nprow)
        if ((step % nprow) == prow) {
            plan.b_offsets[step] = b_offset;  // 记录 B 中此 panel 的偏移
            b_offset += width;                 // 累加本地 K 维度大小
        }
    }

    plan.a_k_local = a_offset;  // A 矩阵本进程的 K 维度总大小
    plan.b_k_local = b_offset;    // B 矩阵本进程的 K 维度总大小
    return plan;
}

// =============================================================================
// GPU 矩阵内存分配
// =============================================================================
// 在 GPU 上分配矩阵内存，并初始化为 0
// 参数:
//   mat  - 矩阵结构体指针
//   rows - 矩阵行数
//   cols - 矩阵列数
// 注意:
//   - 使用 cudaMalloc 分配设备内存
//   - 使用列优先存储，ld (leading dimension) 设为 rows
//   - 即使 cols=0 也分配至少 1 列避免空指针
// =============================================================================
static void allocate_matrix(LocalMatrix* mat, int64_t rows, int64_t cols) {
    mat->rows = rows;
    mat->cols = cols;
    mat->ld = std::max<int64_t>(rows, 1);  // leading dimension 至少为 1

    const int64_t logical_cols = std::max<int64_t>(cols, 1);  // 确保至少分配 1 列
    const size_t bytes = static_cast<size_t>(mat->ld * logical_cols) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&mat->d_ptr, bytes));     // 分配设备内存
    CUDA_CHECK(cudaMemset(mat->d_ptr, 0, bytes));  // 初始化为 0
}

// =============================================================================
// 释放 GPU 矩阵内存
// =============================================================================
// 释放矩阵占用的 GPU 内存并重置结构体状态
// =============================================================================
static void free_matrix(LocalMatrix* mat) {
    if (mat->d_ptr != nullptr) {
        CUDA_CHECK(cudaFree(mat->d_ptr));  // 释放设备内存
        mat->d_ptr = nullptr;
    }
    mat->rows = 0;
    mat->cols = 0;
    mat->ld = 1;
}

static void copy_host_to_device(
    const std::vector<float>& host,
    const LocalMatrix& device,
    cudaStream_t stream) {
    const size_t elems = static_cast<size_t>(device.rows * device.cols);
    if (elems == 0) {
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(
        device.d_ptr,
        host.data(),
        elems * sizeof(float),
        cudaMemcpyHostToDevice,
        stream));
}

static std::vector<float> copy_device_to_host(const LocalMatrix& device) {
    std::vector<float> host(static_cast<size_t>(device.rows * device.cols), 0.0f);
    if (!host.empty()) {
        CUDA_CHECK(cudaMemcpy(
            host.data(),
            device.d_ptr,
            host.size() * sizeof(float),
            cudaMemcpyDeviceToHost));
    }
    return host;
}

static void fill_local_A_host(
    const std::vector<int64_t>& row_indices,
    const PanelPlan& plan,
    int pcol,
    int npcol,
    std::vector<float>* host_A) {
    const int64_t mloc = static_cast<int64_t>(row_indices.size());
    host_A->assign(static_cast<size_t>(mloc * plan.a_k_local), 0.0f);

    for (int step = 0; step < plan.steps; ++step) {
        if ((step % npcol) != pcol) {
            continue;
        }
        const int64_t k0 = plan.starts[step];
        const int64_t ks = plan.sizes[step];
        const int64_t local_col0 = plan.a_offsets[step];
        for (int64_t kk = 0; kk < ks; ++kk) {
            float* dst_col = host_A->data() + static_cast<size_t>((local_col0 + kk) * mloc);
            const int64_t gk = k0 + kk;
            for (int64_t i = 0; i < mloc; ++i) {
                dst_col[i] = value_a(row_indices[static_cast<size_t>(i)], gk);
            }
        }
    }
}

static void fill_local_B_host(
    const std::vector<int64_t>& col_indices,
    const PanelPlan& plan,
    int prow,
    int nprow,
    std::vector<float>* host_B) {
    const int64_t nloc = static_cast<int64_t>(col_indices.size());
    host_B->assign(static_cast<size_t>(plan.b_k_local * nloc), 0.0f);

    for (int step = 0; step < plan.steps; ++step) {
        if ((step % nprow) != prow) {
            continue;
        }
        const int64_t k0 = plan.starts[step];
        const int64_t ks = plan.sizes[step];
        const int64_t local_row0 = plan.b_offsets[step];
        for (int64_t j = 0; j < nloc; ++j) {
            float* dst_col = host_B->data() + static_cast<size_t>(j * plan.b_k_local + local_row0);
            const int64_t gj = col_indices[static_cast<size_t>(j)];
            for (int64_t kk = 0; kk < ks; ++kk) {
                dst_col[kk] = value_b(k0 + kk, gj);
            }
        }
    }
}

static void fill_local_C_host(
    const std::vector<int64_t>& row_indices,
    const std::vector<int64_t>& col_indices,
    std::vector<float>* host_C) {
    const int64_t mloc = static_cast<int64_t>(row_indices.size());
    const int64_t nloc = static_cast<int64_t>(col_indices.size());
    host_C->assign(static_cast<size_t>(mloc * nloc), 0.0f);

    for (int64_t j = 0; j < nloc; ++j) {
        float* dst_col = host_C->data() + static_cast<size_t>(j * mloc);
        const int64_t gj = col_indices[static_cast<size_t>(j)];
        for (int64_t i = 0; i < mloc; ++i) {
            dst_col[i] = value_c(row_indices[static_cast<size_t>(i)], gj);
        }
    }
}

// =============================================================================
// 初始化进程网格 - 设置 MPI 拓扑和 NCCL 通信器
// =============================================================================
// 该函数完成以下工作：
// 1. 获取 MPI 全局信息 (world_rank, world_size)
// 2. 计算当前进程在 2D 网格中的坐标 (prow, pcol)
// 3. 创建行/列 MPI 子通信器
// 4. 检测并绑定 GPU 设备
// 5. 初始化 NCCL 行/列通信器
// =============================================================================
static void init_proc_grid(ProcGrid* grid, int nprow, int npcol) {
    // 获取 MPI 全局信息
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &grid->world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &grid->world_size));
    g_world_rank = grid->world_rank;  // 更新全局 rank 供错误报告使用

    grid->nprow = nprow;
    grid->npcol = npcol;

    // 验证进程网格参数有效性
    if (nprow <= 0 || npcol <= 0 || nprow * npcol != grid->world_size) {
        if (grid->world_rank == 0) {
            fprintf(stderr, "Invalid process grid: nprow=%d npcol=%d world=%d\n",
                    nprow, npcol, grid->world_size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 计算当前进程在 2D 网格中的坐标
    // rank = prow * npcol + pcol
    grid->prow = grid->world_rank / npcol;  // 行坐标
    grid->pcol = grid->world_rank % npcol;  // 列坐标

    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, grid->prow, grid->pcol, &grid->row_comm));
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, grid->pcol, grid->prow, &grid->col_comm));
    MPI_CHECK(MPI_Comm_rank(grid->row_comm, &grid->row_rank));
    MPI_CHECK(MPI_Comm_rank(grid->col_comm, &grid->col_rank));

    MPI_CHECK(MPI_Comm_split_type(
        MPI_COMM_WORLD,
        MPI_COMM_TYPE_SHARED,
        grid->world_rank,
        MPI_INFO_NULL,
        &grid->shared_comm));
    MPI_CHECK(MPI_Comm_rank(grid->shared_comm, &grid->local_rank));
    MPI_CHECK(MPI_Comm_size(grid->shared_comm, &grid->local_size));

    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0) {
        fprintf(stderr, "[rank %d] No visible CUDA devices found.\n", grid->world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (grid->local_rank >= dev_count) {
        if (grid->world_rank == 0) {
            fprintf(stderr,
                    "This demo expects at most one MPI rank per GPU on each node. "
                    "local_rank=%d visible_gpus=%d\n",
                    grid->local_rank,
                    dev_count);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    grid->device = grid->local_rank;
    CUDA_CHECK(cudaSetDevice(grid->device));

    ncclUniqueId row_id;
    if (grid->row_rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&row_id));
    }
    MPI_CHECK(MPI_Bcast(&row_id, sizeof(row_id), MPI_BYTE, 0, grid->row_comm));
    NCCL_CHECK(ncclCommInitRank(&grid->row_nccl, npcol, row_id, grid->row_rank));

    ncclUniqueId col_id;
    if (grid->col_rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&col_id));
    }
    MPI_CHECK(MPI_Bcast(&col_id, sizeof(col_id), MPI_BYTE, 0, grid->col_comm));
    NCCL_CHECK(ncclCommInitRank(&grid->col_nccl, nprow, col_id, grid->col_rank));
}

static void destroy_proc_grid(ProcGrid* grid) {
    if (grid->row_nccl != nullptr) {
        NCCL_CHECK(ncclCommDestroy(grid->row_nccl));
        grid->row_nccl = nullptr;
    }
    if (grid->col_nccl != nullptr) {
        NCCL_CHECK(ncclCommDestroy(grid->col_nccl));
        grid->col_nccl = nullptr;
    }
    if (grid->row_comm != MPI_COMM_NULL) {
        MPI_CHECK(MPI_Comm_free(&grid->row_comm));
        grid->row_comm = MPI_COMM_NULL;
    }
    if (grid->col_comm != MPI_COMM_NULL) {
        MPI_CHECK(MPI_Comm_free(&grid->col_comm));
        grid->col_comm = MPI_COMM_NULL;
    }
    if (grid->shared_comm != MPI_COMM_NULL) {
        MPI_CHECK(MPI_Comm_free(&grid->shared_comm));
        grid->shared_comm = MPI_COMM_NULL;
    }
}

static void scale_or_zero_C(
    LocalMatrix* C,
    float beta,
    cublasHandle_t cublas,
    cudaStream_t stream) {
    const int64_t elems64 = C->rows * C->cols;
    if (elems64 <= 0) {
        return;
    }
    if (beta == 0.0f) {
        CUDA_CHECK(cudaMemsetAsync(
            C->d_ptr,
            0,
            static_cast<size_t>(elems64) * sizeof(float),
            stream));
        return;
    }
    if (beta == 1.0f) {
        return;
    }
    if (elems64 > std::numeric_limits<int>::max()) {
        fprintf(stderr, "[rank %d] Local matrix is too large for cublasSscal.\n", g_world_rank);
        std::abort();
    }
    const int elems = static_cast<int>(elems64);
    CUBLAS_CHECK(cublasSscal(cublas, elems, &beta, C->d_ptr, 1));
}

static void summa_NN_AB_gemm(
    const ProcGrid& grid,
    const PanelPlan& plan,
    float alpha,
    const LocalMatrix& A_local,
    const LocalMatrix& B_local,
    float beta,
    LocalMatrix* C_local,
    cublasHandle_t cublas,
    cudaStream_t stream) {
    scale_or_zero_C(C_local, beta, cublas, stream);

    if (alpha == 0.0f || plan.steps == 0) {
        return;
    }

    LocalMatrix A_panel;
    LocalMatrix B_panel;
    allocate_matrix(&A_panel, A_local.rows, std::max<int64_t>(plan.max_panel, 1));
    allocate_matrix(&B_panel, std::max<int64_t>(plan.max_panel, 1), B_local.cols);

    for (int step = 0; step < plan.steps; ++step) {
        const int a_root = step % grid.npcol;
        const int b_root = step % grid.nprow;
        const int64_t panel_k = plan.sizes[step];

        if (panel_k <= 0) {
            continue;
        }

        if (grid.pcol == a_root && A_local.rows > 0) {
            const int64_t a_offset = plan.a_offsets[step];
            const float* src_A = A_local.d_ptr + a_offset * A_local.ld;
            CUDA_CHECK(cudaMemcpy2DAsync(
                A_panel.d_ptr,
                static_cast<size_t>(A_panel.ld) * sizeof(float),
                src_A,
                static_cast<size_t>(A_local.ld) * sizeof(float),
                static_cast<size_t>(A_local.rows) * sizeof(float),
                static_cast<size_t>(panel_k),
                cudaMemcpyDeviceToDevice,
                stream));
        }

        if (grid.prow == b_root && B_local.cols > 0) {
            const int64_t b_offset = plan.b_offsets[step];
            const float* src_B = B_local.d_ptr + b_offset;
            CUDA_CHECK(cudaMemcpy2DAsync(
                B_panel.d_ptr,
                static_cast<size_t>(panel_k) * sizeof(float),
                src_B,
                static_cast<size_t>(B_local.ld) * sizeof(float),
                static_cast<size_t>(panel_k) * sizeof(float),
                static_cast<size_t>(B_local.cols),
                cudaMemcpyDeviceToDevice,
                stream));
        }

        const size_t a_count = static_cast<size_t>(A_local.rows * panel_k);
        const size_t b_count = static_cast<size_t>(panel_k * B_local.cols);

        if (a_count > 0 || b_count > 0) {
            NCCL_CHECK(ncclGroupStart());
            if (a_count > 0) {
                NCCL_CHECK(ncclBroadcast(
                    A_panel.d_ptr,
                    A_panel.d_ptr,
                    a_count,
                    ncclFloat32,
                    a_root,
                    grid.row_nccl,
                    stream));
            }
            if (b_count > 0) {
                NCCL_CHECK(ncclBroadcast(
                    B_panel.d_ptr,
                    B_panel.d_ptr,
                    b_count,
                    ncclFloat32,
                    b_root,
                    grid.col_nccl,
                    stream));
            }
            NCCL_CHECK(ncclGroupEnd());
        }

        if (C_local->rows > 0 && C_local->cols > 0) {
            if (C_local->rows > std::numeric_limits<int>::max() ||
                C_local->cols > std::numeric_limits<int>::max() ||
                panel_k > std::numeric_limits<int>::max()) {
                fprintf(stderr, "[rank %d] Local SGEMM dimensions exceed cuBLAS int API.\n", g_world_rank);
                std::abort();
            }

            const float gemm_beta = 1.0f;
            CUBLAS_CHECK(cublasSgemm(
                cublas,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                static_cast<int>(C_local->rows),
                static_cast<int>(C_local->cols),
                static_cast<int>(panel_k),
                &alpha,
                A_panel.d_ptr,
                static_cast<int>(A_panel.ld),
                B_panel.d_ptr,
                static_cast<int>(panel_k),
                &gemm_beta,
                C_local->d_ptr,
                static_cast<int>(C_local->ld)));
        }
    }

    free_matrix(&A_panel);
    free_matrix(&B_panel);
}

static double compute_local_abs_sum(cublasHandle_t cublas, const LocalMatrix& mat) {
    const int64_t elems64 = mat.rows * mat.cols;
    if (elems64 <= 0) {
        return 0.0;
    }
    if (elems64 > std::numeric_limits<int>::max()) {
        fprintf(stderr, "[rank %d] Local matrix is too large for cublasSasum.\n", g_world_rank);
        std::abort();
    }
    float local_sum = 0.0f;
    CUBLAS_CHECK(cublasSasum(cublas, static_cast<int>(elems64), mat.d_ptr, 1, &local_sum));
    return static_cast<double>(local_sum);
}

static bool verify_work_too_large(int64_t M, int64_t N, int64_t K, int64_t limit_ops) {
    if (M == 0 || N == 0 || K == 0) {
        return false;
    }
    if (M > limit_ops / N) {
        return true;
    }
    const int64_t mn = M * N;
    return mn > limit_ops / K;
}

static void verify_small_result(
    const ProcGrid& grid,
    const LocalMatrix& C_local,
    int64_t M,
    int64_t N,
    int64_t K,
    int mb,
    int nb,
    float alpha,
    float beta,
    int check_result) {
    if (check_result == 0) {
        return;
    }

    const int64_t max_verify_ops = 256LL * 1024LL * 1024LL;
    if (verify_work_too_large(M, N, K, max_verify_ops)) {
        if (grid.world_rank == 0) {
            printf("Verification skipped: CPU reference would be too large for this problem size.\n");
        }
        return;
    }

    const std::vector<float> h_local_C = copy_device_to_host(C_local);
    const int local_count = static_cast<int>(h_local_C.size());

    std::vector<int> recvcounts;
    std::vector<int> displs;
    std::vector<float> gathered;

    if (grid.world_rank == 0) {
        recvcounts.resize(grid.world_size);
        displs.resize(grid.world_size);
        int disp = 0;
        for (int rank = 0; rank < grid.world_size; ++rank) {
            const int prow = rank / grid.npcol;
            const int pcol = rank % grid.npcol;
            const std::vector<int64_t> row_idx = build_block_cyclic_indices(M, mb, prow, grid.nprow);
            const std::vector<int64_t> col_idx = build_block_cyclic_indices(N, nb, pcol, grid.npcol);
            const int count = static_cast<int>(row_idx.size() * col_idx.size());
            recvcounts[rank] = count;
            displs[rank] = disp;
            disp += count;
        }
        gathered.resize(static_cast<size_t>(disp), 0.0f);
    }

    MPI_CHECK(MPI_Gatherv(
        local_count > 0 ? const_cast<float*>(h_local_C.data()) : nullptr,
        local_count,
        MPI_FLOAT,
        grid.world_rank == 0 ? gathered.data() : nullptr,
        grid.world_rank == 0 ? recvcounts.data() : nullptr,
        grid.world_rank == 0 ? displs.data() : nullptr,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD));

    if (grid.world_rank != 0) {
        return;
    }

    std::vector<float> global_C(static_cast<size_t>(M * N), 0.0f);

    for (int rank = 0; rank < grid.world_size; ++rank) {
        const int prow = rank / grid.npcol;
        const int pcol = rank % grid.npcol;
        const std::vector<int64_t> row_idx = build_block_cyclic_indices(M, mb, prow, grid.nprow);
        const std::vector<int64_t> col_idx = build_block_cyclic_indices(N, nb, pcol, grid.npcol);
        const float* src = gathered.data() + displs[rank];

        for (size_t lj = 0; lj < col_idx.size(); ++lj) {
            const float* src_col = src + lj * row_idx.size();
            float* dst_col = global_C.data() + static_cast<size_t>(col_idx[lj] * M);
            for (size_t li = 0; li < row_idx.size(); ++li) {
                dst_col[row_idx[li]] = src_col[li];
            }
        }
    }

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;

    for (int64_t j = 0; j < N; ++j) {
        for (int64_t i = 0; i < M; ++i) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                acc += static_cast<double>(value_a(i, k)) * static_cast<double>(value_b(k, j));
            }
            const double ref = static_cast<double>(alpha) * acc
                             + static_cast<double>(beta) * static_cast<double>(value_c(i, j));
            const double got = static_cast<double>(global_C[static_cast<size_t>(i + j * M)]);
            const double abs_err = std::fabs(got - ref);
            const double rel_err = abs_err / std::max(1.0, std::fabs(ref));
            max_abs_err = std::max(max_abs_err, abs_err);
            max_rel_err = std::max(max_rel_err, rel_err);
        }
    }

    printf("Verification: max_abs_err=%.6e max_rel_err=%.6e\n", max_abs_err, max_rel_err);
}

static void print_usage(int rank) {
    if (rank != 0) {
        return;
    }
    printf("Usage:\n");
    printf("  mpirun -np <nprow*npcol> ./summa_gemm "
           "<nprow> <npcol> <M> <N> <K> <mb> <nb> <kb> [alpha] [beta] [check]\n");
    printf("\nExample for 4x4090 on a 2x2 grid:\n");
    printf("  mpirun -np 4 ./summa_gemm 2 2 8192 8192 8192 1024 1024 1024 1.0 0.0 0\n");
    printf("  mpirun -np 4 ./summa_gemm 2 2 512 512 512 128 128 128 1.0 0.0 1\n");
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &g_world_rank));

    int world_size = 0;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    int nprow = 2;
    int npcol = 2;
    int64_t M = 8192;
    int64_t N = 8192;
    int64_t K = 8192;
    int mb = 1024;
    int nb = 1024;
    int kb = 1024;
    float alpha = 1.0f;
    float beta = 0.0f;
    int check_result = 0;

    if (argc > 1 && std::strcmp(argv[1], "--help") == 0) {
        print_usage(g_world_rank);
        MPI_CHECK(MPI_Finalize());
        return 0;
    }

    if (argc > 1) nprow = std::atoi(argv[1]);
    if (argc > 2) npcol = std::atoi(argv[2]);
    if (argc > 3) M = std::atoll(argv[3]);
    if (argc > 4) N = std::atoll(argv[4]);
    if (argc > 5) K = std::atoll(argv[5]);
    if (argc > 6) mb = std::atoi(argv[6]);
    if (argc > 7) nb = std::atoi(argv[7]);
    if (argc > 8) kb = std::atoi(argv[8]);
    if (argc > 9) alpha = static_cast<float>(std::atof(argv[9]));
    if (argc > 10) beta = static_cast<float>(std::atof(argv[10]));
    if (argc > 11) check_result = std::atoi(argv[11]);

    if (nprow <= 0 || npcol <= 0 || mb <= 0 || nb <= 0 || kb <= 0 || M < 0 || N < 0 || K < 0) {
        if (g_world_rank == 0) {
            fprintf(stderr, "Matrix dimensions must be non-negative and block sizes must be positive.\n");
            print_usage(g_world_rank);
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    if (nprow * npcol != world_size) {
        if (g_world_rank == 0) {
            fprintf(stderr, "nprow(%d) * npcol(%d) must equal MPI world size(%d).\n",
                    nprow, npcol, world_size);
            print_usage(g_world_rank);
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    ProcGrid grid;
    init_proc_grid(&grid, nprow, npcol);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    const std::vector<int64_t> row_indices = build_block_cyclic_indices(M, mb, grid.prow, grid.nprow);
    const std::vector<int64_t> col_indices = build_block_cyclic_indices(N, nb, grid.pcol, grid.npcol);
    const PanelPlan plan = build_panel_plan(K, kb, grid.prow, grid.nprow, grid.pcol, grid.npcol);

    LocalMatrix A_local;
    LocalMatrix B_local;
    LocalMatrix C_local;
    allocate_matrix(&A_local, static_cast<int64_t>(row_indices.size()), plan.a_k_local);
    allocate_matrix(&B_local, plan.b_k_local, static_cast<int64_t>(col_indices.size()));
    allocate_matrix(&C_local, static_cast<int64_t>(row_indices.size()), static_cast<int64_t>(col_indices.size()));

    std::vector<float> h_A;
    std::vector<float> h_B;
    std::vector<float> h_C;
    fill_local_A_host(row_indices, plan, grid.pcol, grid.npcol, &h_A);
    fill_local_B_host(col_indices, plan, grid.prow, grid.nprow, &h_B);
    fill_local_C_host(row_indices, col_indices, &h_C);

    copy_host_to_device(h_A, A_local, stream);
    copy_host_to_device(h_B, B_local, stream);
    copy_host_to_device(h_C, C_local, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    long long local_shape[7] = {
        static_cast<long long>(A_local.rows),
        static_cast<long long>(A_local.cols),
        static_cast<long long>(B_local.rows),
        static_cast<long long>(B_local.cols),
        static_cast<long long>(C_local.rows),
        static_cast<long long>(C_local.cols),
        static_cast<long long>(grid.device)
    };
    std::vector<long long> all_shapes;
    if (grid.world_rank == 0) {
        all_shapes.resize(static_cast<size_t>(7 * grid.world_size), 0);
    }
    MPI_CHECK(MPI_Gather(
        local_shape,
        7,
        MPI_LONG_LONG,
        grid.world_rank == 0 ? all_shapes.data() : nullptr,
        7,
        MPI_LONG_LONG,
        0,
        MPI_COMM_WORLD));

    if (grid.world_rank == 0) {
        printf("========================================\n");
        printf("SUMMA_NN_AB_GEMM (MPI + NCCL + cuBLAS)\n");
        printf("========================================\n");
        printf("Process grid : %d x %d = %d ranks\n", nprow, npcol, world_size);
        printf("Matrix size  : A(%lld x %lld) * B(%lld x %lld) -> C(%lld x %lld)\n",
               static_cast<long long>(M), static_cast<long long>(K),
               static_cast<long long>(K), static_cast<long long>(N),
               static_cast<long long>(M), static_cast<long long>(N));
        printf("Block sizes  : mb=%d nb=%d kb=%d\n", mb, nb, kb);
        printf("Scalars      : alpha=%g beta=%g\n", alpha, beta);
        printf("Verification : %s\n", check_result ? "enabled" : "disabled");
        printf("----------------------------------------\n");
        for (int rank = 0; rank < world_size; ++rank) {
            const int prow = rank / npcol;
            const int pcol = rank % npcol;
            const long long* shape = all_shapes.data() + 7 * rank;
            printf("rank %d -> grid(%d,%d) gpu=%lld  A(%lld x %lld)  B(%lld x %lld)  C(%lld x %lld)\n",
                   rank, prow, pcol,
                   shape[6], shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
        }
        printf("========================================\n");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double t0 = MPI_Wtime();

    summa_NN_AB_gemm(
        grid,
        plan,
        alpha,
        A_local,
        B_local,
        beta,
        &C_local,
        cublas,
        stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double local_elapsed = MPI_Wtime() - t0;

    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(
        &local_elapsed,
        &elapsed,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD));

    double local_abs_sum = compute_local_abs_sum(cublas, C_local);
    double global_abs_sum = 0.0;
    MPI_CHECK(MPI_Allreduce(
        &local_abs_sum,
        &global_abs_sum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD));

    if (grid.world_rank == 0) {
        const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        const double gflops = flops / (elapsed * 1.0e9);
        printf("Results:\n");
        printf("  time(s)        : %.6f\n", elapsed);
        printf("  total GFLOPS   : %.2f\n", gflops);
        printf("  per-GPU GFLOPS : %.2f\n", gflops / world_size);
        printf("  |C|_1 checksum : %.6e\n", global_abs_sum);
    }

    verify_small_result(
        grid,
        C_local,
        M,
        N,
        K,
        mb,
        nb,
        alpha,
        beta,
        check_result);

    free_matrix(&A_local);
    free_matrix(&B_local);
    free_matrix(&C_local);

    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDA_CHECK(cudaStreamDestroy(stream));
    destroy_proc_grid(&grid);

    MPI_CHECK(MPI_Finalize());
    return 0;
}
