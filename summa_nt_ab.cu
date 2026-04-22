// =============================================================================
// SUMMA_NT_AB_GEMM - distributed complex GEMM
// =============================================================================
// Computes:
//   C = alpha * op(A) * op(B) + beta * C
// where:
//   op(A) = A,    A is stored/distributed in raw shape (M x K)
//   op(B) = B^T,  B is stored/distributed in raw shape (N x K)
//
// Algorithm:
//   - SUMMA broadcast-broadcast variant
//   - A K-panels are broadcast within each process row
//   - B K-panels are broadcast within each process column
//   - local GEMM uses cuBLAS with transa=N, transb=T
//
// Supported types:
//   - cuComplex
//   - cuDoubleComplex
//
// Recommended build (Linux/WSL2):
//   nvcc -O3 -std=c++17 -ccbin mpicxx \
//       -gencode arch=compute_89,code=sm_89 \
//       -o summa_nt_ab summa_nt_ab.cu \
//       -lcublas -lnccl
//
// Example:
//   mpirun -np 4 ./summa_nt_ab 2 2 4096 4096 4096 1024 1024 1024 cfloat 1 0 0 0 1
// =============================================================================

#if defined(_WIN32) && !defined(__linux__)
#error "NCCL is supported on Linux/WSL2 rather than native Windows. Build this demo on Ubuntu/WSL2 on the 4090 machine."
#endif

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static int g_world_rank = -1;

#define MPI_CHECK(cmd) do { \
    const int status_ = (cmd); \
    if (status_ != MPI_SUCCESS) { \
        fprintf(stderr, "[rank %d] MPI error at %s:%d, code=%d\n", \
                g_world_rank, __FILE__, __LINE__, status_); \
        std::abort(); \
    } \
} while (0)

#define CUDA_CHECK(cmd) do { \
    const cudaError_t status_ = (cmd); \
    if (status_ != cudaSuccess) { \
        fprintf(stderr, "[rank %d] CUDA error at %s:%d: %s\n", \
                g_world_rank, __FILE__, __LINE__, cudaGetErrorString(status_)); \
        std::abort(); \
    } \
} while (0)

#define NCCL_CHECK(cmd) do { \
    const ncclResult_t status_ = (cmd); \
    if (status_ != ncclSuccess) { \
        fprintf(stderr, "[rank %d] NCCL error at %s:%d: %s\n", \
                g_world_rank, __FILE__, __LINE__, ncclGetErrorString(status_)); \
        std::abort(); \
    } \
} while (0)

#define CUBLAS_CHECK(cmd) do { \
    const cublasStatus_t status_ = (cmd); \
    if (status_ != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[rank %d] cuBLAS error at %s:%d: %d\n", \
                g_world_rank, __FILE__, __LINE__, static_cast<int>(status_)); \
        std::abort(); \
    } \
} while (0)

struct ProcGrid {
    int world_rank = 0;
    int world_size = 0;

    int nprow = 1;
    int npcol = 1;

    int prow = 0;
    int pcol = 0;

    int row_rank = 0;
    int col_rank = 0;

    int local_rank = 0;
    int local_size = 1;
    int device = 0;

    MPI_Comm row_comm = MPI_COMM_NULL;
    MPI_Comm col_comm = MPI_COMM_NULL;
    MPI_Comm shared_comm = MPI_COMM_NULL;

    ncclComm_t row_nccl = nullptr;
    ncclComm_t col_nccl = nullptr;
};

template <typename T>
struct LocalMatrix {
    T* d_ptr = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t ld = 1;
};

struct PanelPlan {
    int64_t K = 0;
    int kb = 0;
    int steps = 0;
    int64_t max_panel = 0;

    int64_t a_k_local = 0;
    int64_t b_k_local = 0;

    std::vector<int64_t> starts;
    std::vector<int64_t> sizes;
    std::vector<int64_t> a_offsets;
    std::vector<int64_t> b_offsets;
};

template <typename T>
struct ComplexTraits;

template <>
struct ComplexTraits<cuComplex> {
    using Scalar = cuComplex;
    using Real = float;

    static Scalar make(Real re, Real im) {
        return make_cuComplex(re, im);
    }

    static Scalar zero() {
        return make(0.0f, 0.0f);
    }

    static Scalar one() {
        return make(1.0f, 0.0f);
    }

    static bool is_zero(const Scalar& x) {
        return cuCrealf(x) == 0.0f && cuCimagf(x) == 0.0f;
    }

    static bool is_one(const Scalar& x) {
        return cuCrealf(x) == 1.0f && cuCimagf(x) == 0.0f;
    }

    static std::complex<Real> to_std(const Scalar& x) {
        return std::complex<Real>(cuCrealf(x), cuCimagf(x));
    }

    static ncclDataType_t nccl_base_type() {
        return ncclFloat32;
    }

    static size_t nccl_count(size_t complex_count) {
        return complex_count * 2;
    }

    static cublasStatus_t scal(cublasHandle_t handle, int n, const Scalar* alpha, Scalar* x) {
        return cublasCscal(handle, n, alpha, x, 1);
    }

    static cublasStatus_t gemm(
        cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        const Scalar* alpha,
        const Scalar* A,
        int lda,
        const Scalar* B,
        int ldb,
        const Scalar* beta,
        Scalar* C,
        int ldc) {
        return cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    static cublasStatus_t asum(cublasHandle_t handle, int n, const Scalar* x, Real* result) {
        return cublasScasum(handle, n, x, 1, result);
    }

    static const char* name() {
        return "cfloat";
    }
};

template <>
struct ComplexTraits<cuDoubleComplex> {
    using Scalar = cuDoubleComplex;
    using Real = double;

    static Scalar make(Real re, Real im) {
        return make_cuDoubleComplex(re, im);
    }

    static Scalar zero() {
        return make(0.0, 0.0);
    }

    static Scalar one() {
        return make(1.0, 0.0);
    }

    static bool is_zero(const Scalar& x) {
        return cuCreal(x) == 0.0 && cuCimag(x) == 0.0;
    }

    static bool is_one(const Scalar& x) {
        return cuCreal(x) == 1.0 && cuCimag(x) == 0.0;
    }

    static std::complex<Real> to_std(const Scalar& x) {
        return std::complex<Real>(cuCreal(x), cuCimag(x));
    }

    static ncclDataType_t nccl_base_type() {
        return ncclFloat64;
    }

    static size_t nccl_count(size_t complex_count) {
        return complex_count * 2;
    }

    static cublasStatus_t scal(cublasHandle_t handle, int n, const Scalar* alpha, Scalar* x) {
        return cublasZscal(handle, n, alpha, x, 1);
    }

    static cublasStatus_t gemm(
        cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        const Scalar* alpha,
        const Scalar* A,
        int lda,
        const Scalar* B,
        int ldb,
        const Scalar* beta,
        Scalar* C,
        int ldc) {
        return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    static cublasStatus_t asum(cublasHandle_t handle, int n, const Scalar* x, Real* result) {
        return cublasDzasum(handle, n, x, 1, result);
    }

    static const char* name() {
        return "cdouble";
    }
};

static inline int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

static std::vector<int64_t> build_block_cyclic_indices(
    int64_t n,
    int64_t block,
    int owner,
    int nowners) {
    std::vector<int64_t> indices;
    if (n <= 0 || block <= 0 || nowners <= 0) {
        return indices;
    }

    const int64_t nblocks = ceil_div(n, block);
    for (int64_t blk = owner; blk < nblocks; blk += nowners) {
        const int64_t begin = blk * block;
        const int64_t width = std::min<int64_t>(block, n - begin);
        for (int64_t x = 0; x < width; ++x) {
            indices.push_back(begin + x);
        }
    }
    return indices;
}

static PanelPlan build_panel_plan(
    int64_t K,
    int kb,
    int prow,
    int nprow,
    int pcol,
    int npcol) {
    PanelPlan plan;
    plan.K = K;
    plan.kb = kb;
    plan.steps = static_cast<int>(ceil_div(K, static_cast<int64_t>(kb)));
    plan.starts.resize(plan.steps);
    plan.sizes.resize(plan.steps);
    plan.a_offsets.resize(plan.steps, -1);
    plan.b_offsets.resize(plan.steps, -1);

    int64_t a_offset = 0;
    int64_t b_offset = 0;

    for (int step = 0; step < plan.steps; ++step) {
        const int64_t begin = static_cast<int64_t>(step) * kb;
        const int64_t width = std::min<int64_t>(kb, K - begin);
        plan.starts[step] = begin;
        plan.sizes[step] = width;
        plan.max_panel = std::max(plan.max_panel, width);

        if ((step % npcol) == pcol) {
            plan.a_offsets[step] = a_offset;
            a_offset += width;
        }
        if ((step % nprow) == prow) {
            plan.b_offsets[step] = b_offset;
            b_offset += width;
        }
    }

    plan.a_k_local = a_offset;
    plan.b_k_local = b_offset;
    return plan;
}

template <typename T>
static T value_a(int64_t i, int64_t k) {
    using Traits = ComplexTraits<T>;
    using Real = typename Traits::Real;
    const Real re = static_cast<Real>(1.0)
                  + static_cast<Real>(0.0013) * static_cast<Real>((i % 97) - 48)
                  + static_cast<Real>(0.0007) * static_cast<Real>((k % 89) - 44);
    const Real im = static_cast<Real>(-0.25)
                  + static_cast<Real>(0.0011) * static_cast<Real>((i % 73) - 36)
                  - static_cast<Real>(0.0006) * static_cast<Real>((k % 61) - 30);
    return Traits::make(re, im);
}

template <typename T>
static T value_b(int64_t j, int64_t k) {
    using Traits = ComplexTraits<T>;
    using Real = typename Traits::Real;
    const Real re = static_cast<Real>(-0.5)
                  + static_cast<Real>(0.0010) * static_cast<Real>((k % 83) - 41)
                  + static_cast<Real>(0.0009) * static_cast<Real>((j % 79) - 39);
    const Real im = static_cast<Real>(0.75)
                  - static_cast<Real>(0.0008) * static_cast<Real>((k % 67) - 33)
                  + static_cast<Real>(0.0005) * static_cast<Real>((j % 59) - 29);
    return Traits::make(re, im);
}

template <typename T>
static T value_c(int64_t i, int64_t j) {
    using Traits = ComplexTraits<T>;
    using Real = typename Traits::Real;
    const Real re = static_cast<Real>(0.01) * static_cast<Real>((i + 3 * j) % 23);
    const Real im = static_cast<Real>(-0.02) * static_cast<Real>((2 * i + j) % 17);
    return Traits::make(re, im);
}

template <typename T>
static void allocate_matrix(LocalMatrix<T>* mat, int64_t rows, int64_t cols) {
    mat->rows = rows;
    mat->cols = cols;
    mat->ld = std::max<int64_t>(rows, 1);

    const int64_t logical_cols = std::max<int64_t>(cols, 1);
    const size_t bytes = static_cast<size_t>(mat->ld * logical_cols) * sizeof(T);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mat->d_ptr), bytes));
    CUDA_CHECK(cudaMemset(mat->d_ptr, 0, bytes));
}

template <typename T>
static void free_matrix(LocalMatrix<T>* mat) {
    if (mat->d_ptr != nullptr) {
        CUDA_CHECK(cudaFree(mat->d_ptr));
        mat->d_ptr = nullptr;
    }
    mat->rows = 0;
    mat->cols = 0;
    mat->ld = 1;
}

template <typename T>
static void copy_host_to_device(
    const std::vector<T>& host,
    const LocalMatrix<T>& device,
    cudaStream_t stream) {
    const size_t elems = static_cast<size_t>(device.rows * device.cols);
    if (elems == 0) {
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(
        device.d_ptr,
        host.data(),
        elems * sizeof(T),
        cudaMemcpyHostToDevice,
        stream));
}

template <typename T>
static std::vector<T> copy_device_to_host(const LocalMatrix<T>& device) {
    std::vector<T> host(static_cast<size_t>(device.rows * device.cols), ComplexTraits<T>::zero());
    if (!host.empty()) {
        CUDA_CHECK(cudaMemcpy(
            host.data(),
            device.d_ptr,
            host.size() * sizeof(T),
            cudaMemcpyDeviceToHost));
    }
    return host;
}

template <typename T>
static void fill_local_A_host(
    const std::vector<int64_t>& row_indices,
    const PanelPlan& plan,
    int pcol,
    int npcol,
    std::vector<T>* host_A) {
    using Traits = ComplexTraits<T>;
    const int64_t mloc = static_cast<int64_t>(row_indices.size());
    host_A->assign(static_cast<size_t>(plan.a_k_local * mloc), Traits::zero());

    for (int step = 0; step < plan.steps; ++step) {
        if ((step % npcol) != pcol) {
            continue;
        }
        const int64_t k0 = plan.starts[step];
        const int64_t ks = plan.sizes[step];
        const int64_t local_col0 = plan.a_offsets[step];

        for (int64_t kk = 0; kk < ks; ++kk) {
            T* dst_col = host_A->data() + static_cast<size_t>((local_col0 + kk) * mloc);
            const int64_t gk = k0 + kk;
            for (int64_t i = 0; i < mloc; ++i) {
                dst_col[i] = value_a<T>(row_indices[static_cast<size_t>(i)], gk);
            }
        }
    }
}

template <typename T>
static void fill_local_B_host(
    const std::vector<int64_t>& col_indices,
    const PanelPlan& plan,
    int prow,
    int nprow,
    std::vector<T>* host_B) {
    using Traits = ComplexTraits<T>;
    const int64_t nloc = static_cast<int64_t>(col_indices.size());
    host_B->assign(static_cast<size_t>(plan.b_k_local * nloc), Traits::zero());

    for (int step = 0; step < plan.steps; ++step) {
        if ((step % nprow) != prow) {
            continue;
        }
        const int64_t k0 = plan.starts[step];
        const int64_t ks = plan.sizes[step];
        const int64_t local_col0 = plan.b_offsets[step];

        for (int64_t kk = 0; kk < ks; ++kk) {
            T* dst_col = host_B->data() + static_cast<size_t>((local_col0 + kk) * nloc);
            const int64_t gk = k0 + kk;
            for (int64_t j = 0; j < nloc; ++j) {
                dst_col[j] = value_b<T>(col_indices[static_cast<size_t>(j)], gk);
            }
        }
    }
}

template <typename T>
static void fill_local_C_host(
    const std::vector<int64_t>& row_indices,
    const std::vector<int64_t>& col_indices,
    std::vector<T>* host_C) {
    const int64_t mloc = static_cast<int64_t>(row_indices.size());
    const int64_t nloc = static_cast<int64_t>(col_indices.size());
    host_C->assign(static_cast<size_t>(mloc * nloc), ComplexTraits<T>::zero());

    for (int64_t j = 0; j < nloc; ++j) {
        T* dst_col = host_C->data() + static_cast<size_t>(j * mloc);
        const int64_t gj = col_indices[static_cast<size_t>(j)];
        for (int64_t i = 0; i < mloc; ++i) {
            dst_col[i] = value_c<T>(row_indices[static_cast<size_t>(i)], gj);
        }
    }
}

static void init_proc_grid(ProcGrid* grid, int nprow, int npcol) {
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &grid->world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &grid->world_size));
    g_world_rank = grid->world_rank;

    grid->nprow = nprow;
    grid->npcol = npcol;

    if (nprow <= 0 || npcol <= 0 || nprow * npcol != grid->world_size) {
        if (grid->world_rank == 0) {
            fprintf(stderr, "Invalid process grid: nprow=%d npcol=%d world=%d\n",
                    nprow, npcol, grid->world_size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    grid->prow = grid->world_rank / npcol;
    grid->pcol = grid->world_rank % npcol;

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

template <typename T>
static void scale_or_zero_C(
    LocalMatrix<T>* C,
    T beta,
    cublasHandle_t cublas,
    cudaStream_t stream) {
    using Traits = ComplexTraits<T>;
    const int64_t elems64 = C->rows * C->cols;
    if (elems64 <= 0) {
        return;
    }
    if (Traits::is_zero(beta)) {
        CUDA_CHECK(cudaMemsetAsync(
            C->d_ptr,
            0,
            static_cast<size_t>(elems64) * sizeof(T),
            stream));
        return;
    }
    if (Traits::is_one(beta)) {
        return;
    }
    if (elems64 > std::numeric_limits<int>::max()) {
        fprintf(stderr, "[rank %d] Local matrix is too large for cuBLAS scal.\n", g_world_rank);
        std::abort();
    }
    CUBLAS_CHECK(Traits::scal(cublas, static_cast<int>(elems64), &beta, C->d_ptr));
}

template <typename T>
static void summa_NT_AB_gemm(
    const ProcGrid& grid,
    const PanelPlan& plan,
    T alpha,
    const LocalMatrix<T>& A_local,
    const LocalMatrix<T>& B_local,
    T beta,
    LocalMatrix<T>* C_local,
    cublasHandle_t cublas,
    cudaStream_t stream) {
    using Traits = ComplexTraits<T>;

    scale_or_zero_C(C_local, beta, cublas, stream);
    if (Traits::is_zero(alpha) || plan.steps == 0) {
        return;
    }

    LocalMatrix<T> A_panel;
    LocalMatrix<T> B_panel;
    allocate_matrix(&A_panel, C_local->rows, std::max<int64_t>(plan.max_panel, 1));
    allocate_matrix(&B_panel, B_local.rows, std::max<int64_t>(plan.max_panel, 1));

    for (int step = 0; step < plan.steps; ++step) {
        const int a_root = step % grid.npcol;
        const int b_root = step % grid.nprow;
        const int64_t panel_k = plan.sizes[step];

        if (panel_k <= 0) {
            continue;
        }

        if (grid.pcol == a_root && A_local.rows > 0 && A_local.cols > 0) {
            const int64_t a_offset = plan.a_offsets[step];
            const T* src_A = A_local.d_ptr + a_offset * A_local.ld;
            CUDA_CHECK(cudaMemcpy2DAsync(
                A_panel.d_ptr,
                static_cast<size_t>(A_panel.ld) * sizeof(T),
                src_A,
                static_cast<size_t>(A_local.ld) * sizeof(T),
                static_cast<size_t>(A_local.rows) * sizeof(T),
                static_cast<size_t>(panel_k),
                cudaMemcpyDeviceToDevice,
                stream));
        }

        if (grid.prow == b_root && B_local.rows > 0 && B_local.cols > 0) {
            const int64_t b_offset = plan.b_offsets[step];
            const T* src_B = B_local.d_ptr + b_offset * B_local.ld;
            CUDA_CHECK(cudaMemcpy2DAsync(
                B_panel.d_ptr,
                static_cast<size_t>(B_panel.ld) * sizeof(T),
                src_B,
                static_cast<size_t>(B_local.ld) * sizeof(T),
                static_cast<size_t>(B_local.rows) * sizeof(T),
                static_cast<size_t>(panel_k),
                cudaMemcpyDeviceToDevice,
                stream));
        }

        const size_t a_count = static_cast<size_t>(A_local.rows * panel_k);
        const size_t b_count = static_cast<size_t>(B_local.rows * panel_k);

        if (a_count > 0 || b_count > 0) {
            NCCL_CHECK(ncclGroupStart());
            if (a_count > 0) {
                NCCL_CHECK(ncclBroadcast(
                    A_panel.d_ptr,
                    A_panel.d_ptr,
                    Traits::nccl_count(a_count),
                    Traits::nccl_base_type(),
                    a_root,
                    grid.row_nccl,
                    stream));
            }
            if (b_count > 0) {
                NCCL_CHECK(ncclBroadcast(
                    B_panel.d_ptr,
                    B_panel.d_ptr,
                    Traits::nccl_count(b_count),
                    Traits::nccl_base_type(),
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
                fprintf(stderr, "[rank %d] Local GEMM dimensions exceed cuBLAS int API.\n", g_world_rank);
                std::abort();
            }

            const T gemm_beta = Traits::one();
            CUBLAS_CHECK(Traits::gemm(
                cublas,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                static_cast<int>(C_local->rows),
                static_cast<int>(C_local->cols),
                static_cast<int>(panel_k),
                &alpha,
                A_panel.d_ptr,
                static_cast<int>(A_panel.ld),
                B_panel.d_ptr,
                static_cast<int>(B_panel.ld),
                &gemm_beta,
                C_local->d_ptr,
                static_cast<int>(C_local->ld)));
        }
    }

    free_matrix(&A_panel);
    free_matrix(&B_panel);
}

template <typename T>
static double compute_local_abs_sum(cublasHandle_t cublas, const LocalMatrix<T>& mat) {
    using Traits = ComplexTraits<T>;
    using Real = typename Traits::Real;

    const int64_t elems64 = mat.rows * mat.cols;
    if (elems64 <= 0) {
        return 0.0;
    }
    if (elems64 > std::numeric_limits<int>::max()) {
        fprintf(stderr, "[rank %d] Local matrix is too large for cuBLAS asum.\n", g_world_rank);
        std::abort();
    }

    Real local_sum = static_cast<Real>(0);
    CUBLAS_CHECK(Traits::asum(cublas, static_cast<int>(elems64), mat.d_ptr, &local_sum));
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

template <typename T>
static void verify_small_result(
    const ProcGrid& grid,
    const LocalMatrix<T>& C_local,
    int64_t M,
    int64_t N,
    int64_t K,
    int mb,
    int nb,
    T alpha,
    T beta,
    int check_result) {
    using Traits = ComplexTraits<T>;
    using StdComplex = std::complex<typename Traits::Real>;

    if (check_result == 0) {
        return;
    }

    const int64_t max_verify_ops = 128LL * 1024LL * 1024LL;
    if (verify_work_too_large(M, N, K, max_verify_ops)) {
        if (grid.world_rank == 0) {
            printf("Verification skipped: CPU reference would be too large for this problem size.\n");
        }
        return;
    }

    const std::vector<T> h_local_C = copy_device_to_host(C_local);
    const size_t local_bytes64 = h_local_C.size() * sizeof(T);
    const int local_gather_ok =
        (local_bytes64 <= static_cast<size_t>(std::numeric_limits<int>::max())) ? 1 : 0;
    int global_gather_ok = 0;
    MPI_CHECK(MPI_Allreduce(
        &local_gather_ok,
        &global_gather_ok,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD));
    if (global_gather_ok == 0) {
        if (grid.world_rank == 0) {
            printf("Verification skipped: local gather buffer exceeds MPI int count.\n");
        }
        return;
    }
    const int local_bytes = static_cast<int>(local_bytes64);

    std::vector<int> recvcounts;
    std::vector<int> displs;
    std::vector<T> gathered;
    int root_gather_ok = 1;

    if (grid.world_rank == 0) {
        recvcounts.resize(grid.world_size);
        displs.resize(grid.world_size);
        int disp = 0;
        for (int rank = 0; rank < grid.world_size; ++rank) {
            const int prow = rank / grid.npcol;
            const int pcol = rank % grid.npcol;
            const std::vector<int64_t> row_idx = build_block_cyclic_indices(M, mb, prow, grid.nprow);
            const std::vector<int64_t> col_idx = build_block_cyclic_indices(N, nb, pcol, grid.npcol);
            const int64_t count = static_cast<int64_t>(row_idx.size() * col_idx.size());
            const int64_t bytes = count * static_cast<int64_t>(sizeof(T));
            if (bytes > std::numeric_limits<int>::max() ||
                disp > std::numeric_limits<int>::max() - static_cast<int>(bytes)) {
                root_gather_ok = 0;
                break;
            }
            recvcounts[rank] = static_cast<int>(bytes);
            displs[rank] = disp;
            disp += static_cast<int>(bytes);
        }
        if (root_gather_ok != 0) {
            gathered.resize(static_cast<size_t>(disp / static_cast<int>(sizeof(T))), Traits::zero());
        }
    }

    MPI_CHECK(MPI_Bcast(
        &root_gather_ok,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD));
    if (root_gather_ok == 0) {
        if (grid.world_rank == 0) {
            printf("Verification skipped: global gather buffer exceeds MPI int count.\n");
        }
        return;
    }

    MPI_CHECK(MPI_Gatherv(
        local_bytes > 0 ? h_local_C.data() : nullptr,
        local_bytes,
        MPI_BYTE,
        grid.world_rank == 0 && !gathered.empty() ? gathered.data() : nullptr,
        grid.world_rank == 0 ? recvcounts.data() : nullptr,
        grid.world_rank == 0 ? displs.data() : nullptr,
        MPI_BYTE,
        0,
        MPI_COMM_WORLD));

    if (grid.world_rank != 0) {
        return;
    }

    std::vector<T> global_C(static_cast<size_t>(M * N), Traits::zero());

    for (int rank = 0; rank < grid.world_size; ++rank) {
        const int prow = rank / grid.npcol;
        const int pcol = rank % grid.npcol;
        const std::vector<int64_t> row_idx = build_block_cyclic_indices(M, mb, prow, grid.nprow);
        const std::vector<int64_t> col_idx = build_block_cyclic_indices(N, nb, pcol, grid.npcol);
        const unsigned char* gathered_bytes =
            reinterpret_cast<const unsigned char*>(gathered.data());
        const T* src = reinterpret_cast<const T*>(gathered_bytes + displs[rank]);

        for (size_t lj = 0; lj < col_idx.size(); ++lj) {
            const T* src_col = src + lj * row_idx.size();
            T* dst_col = global_C.data() + static_cast<size_t>(col_idx[lj] * M);
            for (size_t li = 0; li < row_idx.size(); ++li) {
                dst_col[row_idx[li]] = src_col[li];
            }
        }
    }

    const StdComplex alpha_h = Traits::to_std(alpha);
    const StdComplex beta_h = Traits::to_std(beta);
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;

    for (int64_t j = 0; j < N; ++j) {
        for (int64_t i = 0; i < M; ++i) {
            StdComplex acc(0.0, 0.0);
            for (int64_t k = 0; k < K; ++k) {
                acc += Traits::to_std(value_a<T>(i, k)) * Traits::to_std(value_b<T>(j, k));
            }
            const StdComplex ref = alpha_h * acc + beta_h * Traits::to_std(value_c<T>(i, j));
            const StdComplex got = Traits::to_std(global_C[static_cast<size_t>(i + j * M)]);
            const double abs_err = std::abs(got - ref);
            const double rel_err = abs_err / std::max(1.0, static_cast<double>(std::abs(ref)));
            max_abs_err = std::max(max_abs_err, abs_err);
            max_rel_err = std::max(max_rel_err, rel_err);
        }
    }

    printf("Verification: max_abs_err=%.6e max_rel_err=%.6e\n", max_abs_err, max_rel_err);
}

enum class RuntimeType {
    kComplexFloat,
    kComplexDouble
};

static RuntimeType parse_runtime_type(const char* arg) {
    if (std::strcmp(arg, "cfloat") == 0 ||
        std::strcmp(arg, "complex64") == 0 ||
        std::strcmp(arg, "cf32") == 0 ||
        std::strcmp(arg, "c") == 0) {
        return RuntimeType::kComplexFloat;
    }
    if (std::strcmp(arg, "cdouble") == 0 ||
        std::strcmp(arg, "complex128") == 0 ||
        std::strcmp(arg, "cf64") == 0 ||
        std::strcmp(arg, "z") == 0) {
        return RuntimeType::kComplexDouble;
    }

    if (g_world_rank == 0) {
        fprintf(stderr, "Unknown dtype '%s'. Use cfloat or cdouble.\n", arg);
    }
    MPI_CHECK(MPI_Finalize());
    std::exit(1);
}

template <typename T>
static void print_scalar(const char* name, T value) {
    const auto z = ComplexTraits<T>::to_std(value);
    printf("%s=(%.6g, %.6g)", name, static_cast<double>(z.real()), static_cast<double>(z.imag()));
}

static void print_usage(int rank) {
    if (rank != 0) {
        return;
    }

    printf("Usage:\n");
    printf("  mpirun -np <nprow*npcol> ./summa_nt_ab "
           "<nprow> <npcol> <M> <N> <K> <mb> <nb> <kb> "
           "[dtype] [alpha_r] [alpha_i] [beta_r] [beta_i] [check]\n");
    printf("\n");
    printf("Arguments:\n");
    printf("  dtype  : cfloat | cdouble (default: cfloat)\n");
    printf("  alpha  : complex scalar, passed as real/imag pair (default: 1 + 0i)\n");
    printf("  beta   : complex scalar, passed as real/imag pair (default: 0 + 0i)\n");
    printf("  check  : 0/1, enable small-size CPU verification (default: 0)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  mpirun -np 4 ./summa_nt_ab 2 2 4096 4096 4096 1024 1024 1024 cfloat 1 0 0 0 0\n");
    printf("  mpirun -np 4 ./summa_nt_ab 2 2 512 512 512 128 128 128 cdouble 1 0 0 0 1\n");
}

template <typename T>
static int run_summa(
    int nprow,
    int npcol,
    int64_t M,
    int64_t N,
    int64_t K,
    int mb,
    int nb,
    int kb,
    T alpha,
    T beta,
    int check_result) {
    using Traits = ComplexTraits<T>;

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

    LocalMatrix<T> A_local;
    LocalMatrix<T> B_local;
    LocalMatrix<T> C_local;
    allocate_matrix(&A_local, static_cast<int64_t>(row_indices.size()), plan.a_k_local);
    allocate_matrix(&B_local, static_cast<int64_t>(col_indices.size()), plan.b_k_local);
    allocate_matrix(&C_local, static_cast<int64_t>(row_indices.size()), static_cast<int64_t>(col_indices.size()));

    std::vector<T> h_A;
    std::vector<T> h_B;
    std::vector<T> h_C;
    fill_local_A_host<T>(row_indices, plan, grid.pcol, grid.npcol, &h_A);
    fill_local_B_host<T>(col_indices, plan, grid.prow, grid.nprow, &h_B);
    fill_local_C_host<T>(row_indices, col_indices, &h_C);

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
        printf("SUMMA_NT_AB_GEMM (MPI + NCCL + cuBLAS)\n");
        printf("========================================\n");
        printf("Process grid : %d x %d = %d ranks\n", nprow, npcol, grid.world_size);
        printf("Data type    : %s\n", Traits::name());
        printf("Matrix size  : A(%lld x %lld) * B^T(from %lld x %lld) -> C(%lld x %lld)\n",
               static_cast<long long>(M), static_cast<long long>(K),
               static_cast<long long>(N), static_cast<long long>(K),
               static_cast<long long>(M), static_cast<long long>(N));
        printf("Block sizes  : mb=%d nb=%d kb=%d\n", mb, nb, kb);
        printf("Scalars      : ");
        print_scalar("alpha", alpha);
        printf("  ");
        print_scalar("beta", beta);
        printf("\n");
        printf("Verification : %s\n", check_result ? "enabled" : "disabled");
        printf("----------------------------------------\n");
        for (int rank = 0; rank < grid.world_size; ++rank) {
            const int prow = rank / npcol;
            const int pcol = rank % npcol;
            const long long* shape = all_shapes.data() + 7 * rank;
            printf("rank %d -> grid(%d,%d) gpu=%lld  A(%lld x %lld)  Braw(%lld x %lld)  C(%lld x %lld)\n",
                   rank,
                   prow,
                   pcol,
                   shape[6],
                   shape[0],
                   shape[1],
                   shape[2],
                   shape[3],
                   shape[4],
                   shape[5]);
        }
        printf("========================================\n");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double t0 = MPI_Wtime();

    summa_NT_AB_gemm(
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
    const double local_elapsed = MPI_Wtime() - t0;

    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(
        &local_elapsed,
        &elapsed,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD));

    const double local_abs_sum = compute_local_abs_sum(cublas, C_local);
    double global_abs_sum = 0.0;
    MPI_CHECK(MPI_Allreduce(
        &local_abs_sum,
        &global_abs_sum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD));

    if (grid.world_rank == 0) {
        const double flops = 8.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        const double gflops = (elapsed > 0.0) ? (flops / (elapsed * 1.0e9)) : 0.0;
        printf("Results:\n");
        printf("  time(s)        : %.6f\n", elapsed);
        printf("  total GFLOPS   : %.2f\n", gflops);
        printf("  per-GPU GFLOPS : %.2f\n", gflops / grid.world_size);
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
    return 0;
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &g_world_rank));

    int world_size = 0;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    int nprow = 2;
    int npcol = 2;
    int64_t M = 4096;
    int64_t N = 4096;
    int64_t K = 4096;
    int mb = 1024;
    int nb = 1024;
    int kb = 1024;
    RuntimeType runtime_type = RuntimeType::kComplexFloat;
    double alpha_r = 1.0;
    double alpha_i = 0.0;
    double beta_r = 0.0;
    double beta_i = 0.0;
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
    if (argc > 9) runtime_type = parse_runtime_type(argv[9]);
    if (argc > 10) alpha_r = std::atof(argv[10]);
    if (argc > 11) alpha_i = std::atof(argv[11]);
    if (argc > 12) beta_r = std::atof(argv[12]);
    if (argc > 13) beta_i = std::atof(argv[13]);
    if (argc > 14) check_result = std::atoi(argv[14]);

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

    int rc = 0;
    if (runtime_type == RuntimeType::kComplexFloat) {
        rc = run_summa(
            nprow,
            npcol,
            M,
            N,
            K,
            mb,
            nb,
            kb,
            ComplexTraits<cuComplex>::make(static_cast<float>(alpha_r), static_cast<float>(alpha_i)),
            ComplexTraits<cuComplex>::make(static_cast<float>(beta_r), static_cast<float>(beta_i)),
            check_result);
    } else {
        rc = run_summa(
            nprow,
            npcol,
            M,
            N,
            K,
            mb,
            nb,
            kb,
            ComplexTraits<cuDoubleComplex>::make(alpha_r, alpha_i),
            ComplexTraits<cuDoubleComplex>::make(beta_r, beta_i),
            check_result);
    }

    MPI_CHECK(MPI_Finalize());
    return rc;
}
