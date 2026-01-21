// gemm_naive.cu
// Naive CUDA GEMM (global memory only) with optional transposeA/transposeB,
// and in-place update: C = alpha * op(A) * op(B) + beta * C
//
// Build (Windows, in a VS Developer Command Prompt / VS Code Developer PowerShell):
//   nvcc -O2 -std=c++17 -allow-unsupported-compiler gemm_naive.cu -o gemm.exe
//
// Run:
//   .\gemm.exe
//
// NOTE: CUDA runs only on NVIDIA GPUs. If you have Intel/AMD GPU only,
// this will print a message and exit.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s (%d) %s\n",               \
                         __FILE__, __LINE__,                                     \
                         cudaGetErrorName(err__), (int)err__,                    \
                         cudaGetErrorString(err__));                             \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

// Row-major indexing helper
__host__ __device__ __forceinline__ int idx2(int r, int c, int ld) {
    return r * ld + c;
}

// A is either (m x k) if !tA, or stored as (k x m) representing A^T if tA
// B is either (k x n) if !tB, or stored as (n x k) representing B^T if tB
__global__ void gemm_naive_kernel(
    int m, int n, int k,
    float alpha,
    const float* __restrict__ A, bool tA,
    const float* __restrict__ B, bool tB,
    float beta,
    float* __restrict__ C
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0..m)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0..n)

    if (row < m && col < n) {
        float acc = 0.0f;

        // dot product over k
        for (int p = 0; p < k; ++p) {
            // op(A)[row, p]
            float a = tA ? A[idx2(p, row, m)]   // A stored as (k x m) == A^T
                         : A[idx2(row, p, k)];  // A stored as (m x k)

            // op(B)[p, col]
            float b = tB ? B[idx2(col, p, k)]   // B stored as (n x k) == B^T
                         : B[idx2(p, col, n)];  // B stored as (k x n)

            acc += a * b;
        }

        float old = C[idx2(row, col, n)];
        C[idx2(row, col, n)] = alpha * acc + beta * old;
    }
}

// CPU reference (same layout rules)
static void gemm_cpu_ref(
    int m, int n, int k,
    float alpha,
    const std::vector<float>& A, bool tA,
    const std::vector<float>& B, bool tB,
    float beta,
    std::vector<float>& C
) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < k; ++p) {
                float a = tA ? A[idx2(p, i, m)] : A[idx2(i, p, k)];
                float b = tB ? B[idx2(j, p, k)] : B[idx2(p, j, n)];
                acc += a * b;
            }
            C[idx2(i, j, n)] = alpha * acc + beta * C[idx2(i, j, n)];
        }
    }
}

static bool nearly_equal(float a, float b, float atol = 1e-3f, float rtol = 1e-3f) {
    float diff = std::fabs(a - b);
    float tol = atol + rtol * std::fabs(b);
    return diff <= tol;
}

static void fill_random(std::vector<float>& v, float lo = -1.0f, float hi = 1.0f, unsigned seed = 123) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : v) x = dist(rng);
}

static void run_one_case(int m, int n, int k, bool tA, bool tB) {
    std::cout << "Case m=" << m << " n=" << n << " k=" << k
              << " tA=" << (tA ? "true" : "false")
              << " tB=" << (tB ? "true" : "false") << "\n";

    float alpha = 1.25f;
    float beta  = 0.75f;

    // Allocate host buffers respecting storage layout:
    // A storage: (!tA) => m*k, (tA) => k*m
    // B storage: (!tB) => k*n, (tB) => n*k
    size_t Asz = (size_t)(tA ? (k * m) : (m * k));
    size_t Bsz = (size_t)(tB ? (n * k) : (k * n));
    size_t Csz = (size_t)(m * n);

    std::vector<float> hA(Asz), hB(Bsz), hC(Csz), hC_ref(Csz);

    fill_random(hA, -1.0f, 1.0f, 111);
    fill_random(hB, -1.0f, 1.0f, 222);
    fill_random(hC, -1.0f, 1.0f, 333);

    hC_ref = hC;

    // CPU reference
    gemm_cpu_ref(m, n, k, alpha, hA, tA, hB, tB, beta, hC_ref);

    // Device alloc
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dA, Asz * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dB, Bsz * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dC, Csz * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), Asz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), Bsz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC.data(), Csz * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x,
                (m + threads.y - 1) / threads.y);

    gemm_naive_kernel<<<blocks, threads>>>(m, n, k, alpha, dA, tA, dB, tB, beta, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, Csz * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare
    int bad = 0;
    float max_abs_err = 0.0f;
    for (int i = 0; i < m * n; ++i) {
        float a = hC[i];
        float b = hC_ref[i];
        float e = std::fabs(a - b);
        max_abs_err = std::max(max_abs_err, e);
        if (!nearly_equal(a, b)) bad++;
    }

    std::cout << "  -> " << (bad == 0 ? "PASS" : "FAIL")
              << " (" << bad << " mismatches, max_abs_err=" << max_abs_err << ")\n\n";

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

int main() {
    // Make startup failure readable on non-NVIDIA machines
    int deviceCount = 0;
    cudaError_t e = cudaGetDeviceCount(&deviceCount);
    if (e != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA-capable NVIDIA GPU found.\n";
        std::cerr << "CUDA requires an NVIDIA GPU + NVIDIA driver.\n";
        std::cerr << "cudaGetDeviceCount: " << cudaGetErrorName(e)
                  << " - " << cudaGetErrorString(e) << "\n";
        return 1;
    }

    // Print device name
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << "\n\n";

    // Test a few cases, including non-multiples of block size
    run_one_case(64, 64, 64, false, false);
    run_one_case(63, 65, 37, false, false);
    run_one_case(32, 48, 16, true,  false);
    run_one_case(32, 48, 16, false, true);
    run_one_case(32, 48, 16, true,  true);

    std::cout << "Done.\n";
    return 0;
}
