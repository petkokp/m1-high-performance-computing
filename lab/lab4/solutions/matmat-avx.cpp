// Results for N=2048, with B1=32 and B2=256:

// Sequential scalar matmat i->j->k took 4.599354e+01s.
// Performance: 373.35 Mflops/s

// Sequential scalar matmat i->k->j took 1.84e+01s.
// Performance: 933.92 Mflops/s

// Single tile scalar matmat i->k->j took 2.29e+01s.
// Performance: 749.52 Mflops/s

// Double tile scalar matmat i->k->j took 2.14e+01s.
// Performance: 803.94 Mflops/s

// Double tile AVX matmat i->k->j took 4.53e+00s.
// Performance: 3791.82 Mflops/s

// Task parallel double tile AVX matmat i->k->j took 6.57e-01s.
// Performance: 26125.54 Mflops/s

// b) The second version is faster because of contiguous memory
// accesses for the B matrix.

// c) The algorithm is much slower with low B1 values  B1 values
// because we do not use the cache as much as we could and
// the creation of the tiles is slow.
// The optimal B1 value based on my L1 and L2 cache is 64.

// g) Speedup for N=2048 is 70. Around 10% of the peak performance of my computer was used.

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cstdlib>
#include "immintrin.h"
#include <chrono>
#include "omp.h"

#define NREPEAT 10

void printUsage(int argc, char **argv)
{
  printf("Usage: %s N\n", argv[0]);
  printf("Example: %s 1024\n", argv[0]);
}

void verify(const float *A, const float *B, const float *C, int N)
{
  int correct = 1;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (C[i * N + j] != N) {
        printf("C(%d, %d) = %f is incorrect; C(%d, %d) should be %d\n", i, j, C[i * N + j], i, j, N);
        correct = 0;
        break;
      }
    }
  }
  if (correct) {
    printf("The result is correct!\n\n");
  } else {
    printf("The result is not correct!\n\n");
  }
}

inline void loadTile(__m256 tile[8], float *addr, int N)
{
  for (int i = 0; i < 8; i ++) {
    tile[i] = _mm256_loadu_ps(&addr[i * N]);
  }
}

inline void storeTile(__m256 tile[8], float *addr, int N)
{
  for (int i = 0; i < 8; i++) {
    _mm256_storeu_ps(&addr[i * N], tile[i]);
  }
}

inline void multiplyTile(float *tA, float *tB, float *tC, __m256 atile[8], __m256 btile[8], __m256 ctile[8], int N)
{
  loadTile(btile, tB, N);
  loadTile(ctile, tC, N);

  for (int i = 0; i < 8; i++) {
    for (int k = 0; k < 8; k++) {
      __m256 a_vec = _mm256_broadcast_ss(&tA[i * N + k]);
      ctile[i] = _mm256_fmadd_ps(a_vec, btile[k], ctile[i]);
    }
  }
  storeTile(ctile, tC, N);
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    printUsage(argc, argv);
    return 0;
  }
  int N = std::atoi(argv[1]);
  const int B1 = 64;
  const int B2 = 512;

  // Allocate and initialize the matrix A and vectors x, b
  float *A = (float *)_mm_malloc(N * N * sizeof(float), 32);
  float *B = (float *)_mm_malloc(N * N * sizeof(float), 32);
  float *C = (float *)_mm_malloc(N * N * sizeof(float), 32);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = 1.0f;
      B[i * N + j] = 1.0f;
      C[i * N + j] = 0.0f;
    }
  }

  // Sequential and scalar matrix-matrix multiplication code with loop order i->j->k
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
          C[i * N + j] += A[i * N + k] * B[k * N + j];
        }
      }
    }
    std::chrono::duration<double> timeDiff = std::chrono::high_resolution_clock::now() - start;
    std::cout << std::scientific << "Sequential scalar matmat i->j->k took " << timeDiff.count() << "s." << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Performance: " << 2.0*N*N*(N-1) / ((1e6) * timeDiff.count()) <<
      "Mflops/s" << std::endl;
    verify(A, B, C, N);
  }

  // Sequential and scalar matrix-matrix multiplication code with loop order i->k->j
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < NREPEAT; repeat++) {
      memset(&C[0], 0, N * N * sizeof(float));
      for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
          for (int j = 0; j < N; j++) {
            C[i * N + j] += A[i * N + k] * B[k * N + j];
          }
        }
      }
    }
    std::chrono::duration<double> timeDiff = (std::chrono::high_resolution_clock::now() - start) / NREPEAT;
    std::cout << std::scientific << "Sequential scalar matmat i->k->j took " << timeDiff.count() << "s." << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Performance: " << 2.0*N*N*(N-1) / ((1e6) * timeDiff.count()) <<
      "Mflops/s" << std::endl;
    verify(A, B, C, N);
  }

  // Sequential and scalar matrix-matrix multiplication code with loop order i->k->j and single level tiling
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < NREPEAT; repeat++) {
      memset(&C[0], 0, N * N * sizeof(float));
      for (int i = 0; i < N; i += B1) {
        for (int k = 0; k < N; k += B1) {
          for (int j = 0; j < N; j += B1) {
            float *tA = &A[i * N + k];
            float *tB = &B[k * N + j];
            float *tC = &C[i * N + j];
            for (int i2 = 0; i2 < B1; i2++) {
              for (int k2 = 0; k2 < B1; k2++) {
                for (int j2 = 0; j2 < B1; j2++) {
                  tC[i2 * N + j2] += tA[i2 * N + k2] * tB[k2 * N + j2];
                }
              }
            }
          }
        }
      }
    }
    std::chrono::duration<double> timeDiff = (std::chrono::high_resolution_clock::now() - start) / NREPEAT;
    std::cout << std::scientific << "Single tile scalar matmat i->k->j took " << timeDiff.count() << "s." << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Performance: " << 2.0*N*N*(N-1) / ((1e6) * timeDiff.count()) <<
      "Mflops/s" << std::endl;
    verify(A, B, C, N);
  }


  // Sequential and scalar matrix-matrix multiplication code with loop order i->k->j and two level tiling
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < NREPEAT; repeat++) {
      memset(&C[0], 0, N * N * sizeof(float));
      for (int i = 0; i < N; i += B2) {
        for (int k = 0; k < N; k += B2) {
          for (int j = 0; j < N; j += B2) {
            for (int i1 = i; i1 < i + B2; i1 += B1) {
              for (int k1 = k; k1 < k + B2; k1 += B1) {
                for (int j1 = j; j1 < j + B2; j1 += B1) {
                  float *tA = &A[i1 * N + k1];
                  float *tB = &B[k1 * N + j1];
                  float *tC = &C[i1 * N + j1];
                  for (int i2 = 0; i2 < B1; i2++) {
                    for (int k2 = 0; k2 < B1; k2++) {
                      for (int j2 = 0; j2 < B1; j2++) {
                        tC[i2 * N + j2] += tA[i2 * N + k2] * tB[k2 * N + j2];
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    std::chrono::duration<double> timeDiff = (std::chrono::high_resolution_clock::now() - start) / NREPEAT;
    std::cout << std::scientific << "Double tile scalar matmat i->k->j took " << timeDiff.count() << "s." << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Performance: " << 2.0*N*N*(N-1) / ((1e6) * timeDiff.count()) <<
      "Mflops/s" << std::endl;
    verify(A, B, C, N);
  }


  // Vectorized matrix-matrix multiplication code with loop order i->k->j and two level tiling + AVX
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < NREPEAT; repeat++) {
      memset(&C[0], 0, N * N * sizeof(float));
      __m256 atile[8], btile[8], ctile[8];
      for (int i = 0; i < N; i += B1) {
        for (int k = 0; k < N; k += B1) {
          for (int j = 0; j < N; j += B1) {
            for (int i1 = 0; i1 < B1; i1 += 8) {
              for (int k1 = 0; k1 < B1; k1 += 8) {
                for (int j1 = 0; j1 < B1; j1 += 8) {
                float *tA = &A[(i + i1) * N + k + k1];
                float *tB = &B[(k + k1) * N + j + j1];
                float *tC = &C[(i + i1) * N + j + j1];
                multiplyTile(tA, tB, tC, atile, btile, ctile, N);
                }
              }
            }
          }
        }
      }
    }
    std::chrono::duration<double> timeDiff = (std::chrono::high_resolution_clock::now() - start) / NREPEAT;
    std::cout << std::scientific << "Double tile AVX matmat i->k->j took " << timeDiff.count() << "s." << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Performance: " << 2.0*N*N*(N-1) / ((1e6) * timeDiff.count()) <<
      "Mflops/s" << std::endl;
    verify(A, B, C, N);
  }

  // Task-parallel and vectorized matrix-matrix multiplication code with loop order i->k->j and two level tiling + AVX
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < NREPEAT; repeat++) {
      memset(&C[0], 0, N * N * sizeof(float));
      #pragma omp parallel num_threads(16)
      {
        __m256 atile[8], btile[8], ctile[8];
        for (int k = 0; k < N; k += B1) {
          #pragma omp for collapse(2)
          for (int i = 0; i < N; i += B1) {
            for (int j = 0; j < N; j += B1) {
              #pragma omp task firstprivate(i, k, j) depend(in: A[i*N + k]) depend(in: B[k*N + j]) depend(out: C[i*N + j])
              for (int i1 = 0; i1 < B1; i1 += 8) {
                for (int k1 = 0; k1 < B1; k1 += 8) {
                  for (int j1 = 0; j1 < B1; j1 += 8) {
                    float *tA = &A[(i + i1) * N + k + k1];
                    float *tB = &B[(k + k1) * N + j + j1];
                    float *tC = &C[(i + i1) * N + j + j1];
                    multiplyTile(tA, tB, tC, atile, btile, ctile, N);
                  }
                }
              }
            }
          }
        }
      }
    }
    std::chrono::duration<double> timeDiff = (std::chrono::high_resolution_clock::now() - start) / NREPEAT;
    std::cout << std::scientific << "Task parallel double tile AVX matmat i->k->j took " << timeDiff.count() << "s." << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "Performance: " << 2.0*N*N*(N-1) / ((1e6) * timeDiff.count()) <<
      "Mflops/s" << std::endl;
    verify(A, B, C, N);
  }

  // Free matrices
  _mm_free(A);
  _mm_free(B);
  _mm_free(C);

  return 0;
}
