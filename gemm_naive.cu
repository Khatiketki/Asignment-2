#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err = (call);                                                    \
    if (err != cudaSuccess) {                                                    \
      std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << ": "          \
                << cudaGetErrorString(err) << "\n";                              \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; // [0..M)
  int col = blockIdx.x * blockDim.x + threadIdx.x; // [0..N)
  if (row < M && col < N) {
    float acc = 0.0f;
    // A is MxK (row-major), B is KxN (row-major), C is MxN
    for (int k = 0; k < K; k++) {
      acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
  }
}

static void cpu_gemm(const std::vector<float>& A,
                     const std::vector<float>& B,
                     std::vector<float>& C,
                     int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float acc = 0.0f;
      for (int k = 0; k < K; k++) {
        acc += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = acc;
    }
  }
}

int main(int argc, char** argv) {
  // Default sizes (you can change or pass args)
  int M = 512, N = 512, K = 512;
  if (argc == 4) {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    K = std::atoi(argv[3]);
  }

  int deviceCount = 0;
  cudaError_t cntErr = cudaGetDeviceCount(&deviceCount);
  if (cntErr != cudaSuccess || deviceCount == 0) {
    std::cerr << "No CUDA-capable NVIDIA GPU found.\n";
    std::cerr << "CUDA requires an NVIDIA GPU + NVIDIA driver.\n";
    std::cerr << "cudaGetDeviceCount: " << cudaGetErrorString(cntErr) << "\n";
    return 1;
  }

  int dev = 0;
  CUDA_CHECK(cudaSetDevice(dev));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::cout << "Using GPU: " << prop.name << "\n";
  std::cout << "GEMM sizes: M=" << M << " N=" << N << " K=" << K << "\n";

  size_t bytesA = (size_t)M * K * sizeof(float);
  size_t bytesB = (size_t)K * N * sizeof(float);
  size_t bytesC = (size_t)M * N * sizeof(float);

  std::vector<float> hA((size_t)M * K);
  std::vector<float> hB((size_t)K * N);
  std::vector<float> hC((size_t)M * N, 0.0f);
  std::vector<float> hCref((size_t)M * N, 0.0f);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : hA) x = dist(rng);
  for (auto& x : hB) x = dist(rng);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, bytesA));
  CUDA_CHECK(cudaMalloc(&dB, bytesB));
  CUDA_CHECK(cudaMalloc(&dC, bytesC));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, bytesC));

  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  // Warmup
  gemm_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  gemm_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));

  // CPU reference check
  cpu_gemm(hA, hB, hCref, M, N, K);

  double maxAbsErr = 0.0;
  for (size_t i = 0; i < hC.size(); i++) {
    double err = std::fabs((double)hC[i] - (double)hCref[i]);
    if (err > maxAbsErr) maxAbsErr = err;
  }

  double flops = 2.0 * (double)M * (double)N * (double)K;
  double gflops = flops / (ms * 1e6);

  std::cout << "Kernel time: " << ms << " ms\n";
  std::cout << "Throughput: " << gflops << " GFLOP/s\n";
  std::cout << "Max abs error vs CPU: " << maxAbsErr << "\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));

  return 0;
}
