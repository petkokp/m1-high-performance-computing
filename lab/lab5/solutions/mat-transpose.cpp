#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cstdlib>
#include "immintrin.h"
#include <chrono>
#include "omp.h"

#define NREPET 1001

void printUsage(int argc, char **argv)
{
  printf("Usage: %s N\n", argv[0]);
  printf("Example: %s 1024\n", argv[0]);
}

inline void loadTile(__m256 tile[], float *addr, int N)
{
  tile[0] = _mm256_load_ps(addr);
  tile[1] = _mm256_load_ps(addr + N);
  tile[2] = _mm256_load_ps(addr + 2 * N);
  tile[3] = _mm256_load_ps(addr + 3 * N);
  tile[4] = _mm256_load_ps(addr + 4 * N);
  tile[5] = _mm256_load_ps(addr + 5 * N);
  tile[6] = _mm256_load_ps(addr + 6 * N);
  tile[7] = _mm256_load_ps(addr + 7 * N);
}

inline void storeTile(__m256 tile[], float *addr, int N)
{
  _mm256_store_ps(addr, tile[0]);
  _mm256_store_ps(addr + N, tile[1]);
  _mm256_store_ps(addr + 2 * N, tile[2]);
  _mm256_store_ps(addr + 3 * N, tile[3]);
  _mm256_store_ps(addr + 4 * N, tile[4]);
  _mm256_store_ps(addr + 5 * N, tile[5]);
  _mm256_store_ps(addr + 6 * N, tile[6]);
  _mm256_store_ps(addr + 7 * N, tile[7]);
}

void printMatrix(const float *mat, int N)
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%3.f ", mat[i * N + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void printTile(__m256 tile[8])
{
  float temp[64] __attribute__((aligned(32)));
  storeTile(tile, temp, 8);
  printMatrix(&temp[0], 8);
}

void verify(const float *mat, int N)
{
  int correct = 1;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (mat[i * N + j] != j - i) {
        printf("mat(%d, %d) = %.f is incorrect; mat(%d, %d) should be %.f\n", i, j, mat[i * N + j], i, j, (float)(j - i));
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


inline void transAVX8x8_ps(__m256 tile[8])
{
  __m256 tile2[8];
  __m256 tile3[8];

  // Compute tile2 from tile using _mm256_unpacklo_ps and _mm256_unpackhi_ps
   for (int i = 0; i < 8; i += 2) {
    tile2[i] = _mm256_unpacklo_ps(tile[i], tile[i+1]);
    tile2[i+1] = _mm256_unpackhi_ps(tile[i], tile[i+1]);
  }
  
  // Compute tile3 from tile2 using _mm256_shuffle_ps
  for (int i = 0; i < 8; i += 4) {
    tile3[i] = _mm256_shuffle_ps(tile2[i], tile2[i+2], _MM_SHUFFLE(1, 0, 1, 0));
    tile3[i+1] = _mm256_shuffle_ps(tile2[i+1], tile2[i+3], _MM_SHUFFLE(1, 0, 1, 0));
    tile3[i+2] = _mm256_shuffle_ps(tile2[i], tile2[i+2], _MM_SHUFFLE(3, 2, 3, 2));
    tile3[i+3] = _mm256_shuffle_ps(tile2[i+1], tile2[i+3], _MM_SHUFFLE(3, 2, 3, 2));
  }

  // Compute tile from tile3 using _mm256_permute2f128_ps
  tile[0] = _mm256_permute2f128_ps(tile3[0], tile3[4], 0x20);
  tile[1] = _mm256_permute2f128_ps(tile3[2], tile3[6], 0x20);
  tile[2] = _mm256_permute2f128_ps(tile3[1], tile3[5], 0x20);
  tile[3] = _mm256_permute2f128_ps(tile3[3], tile3[7], 0x20);
  tile[4] = _mm256_permute2f128_ps(tile3[0], tile3[4], 0x31);
  tile[5] = _mm256_permute2f128_ps(tile3[2], tile3[6], 0x31);
  tile[6] = _mm256_permute2f128_ps(tile3[1], tile3[5], 0x31);
  tile[7] = _mm256_permute2f128_ps(tile3[3], tile3[7], 0x31);
}

// Transposition of an 8x8 submatrix at address tA, and stores it in the 8x8 submatrix at address tB
inline void transAVX8x8(float *tA, float *tB, __m256 tile[8], int N)
{
  loadTile(tile, tA, N);
  transAVX8x8_ps(tile);
  storeTile(tile, tB, N);
}

// In-place transposition of two symmetric 8x8 submatrices at adresses tA and tA2
inline void transAVX8x8InPlace(float *tA, float *tA2, __m256 tile[8], __m256 tile2[8], int N)
{
  // TODO / A FAIRE ...
  loadTile(tile, tA, N);
  loadTile(tile2, tA2, N);
  
  transAVX8x8_ps(tile);
  transAVX8x8_ps(tile2);

  storeTile(tile, tA2, N);
  storeTile(tile2, tA, N);
}

int main(int argc, char **argv)
{
  // Get parameters
  if (argc != 2) {
    printUsage(argc, argv);
    return 0;
  }
  int N = std::atoi(argv[1]);
  __m256 tile[8], tile2[8];

  // Allocate and initialize the matrix A and B
  float *A = (float *)_mm_malloc(N * N * sizeof(float), 32);
  float *B = (float *)_mm_malloc(N * N * sizeof(float), 32);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = i - j;
      B[i * N + j] = 0;
    }
  }

  // Transpose the matrix with a sequential and scalar (non-vectorized) code
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int repet = 0; repet < NREPET; repet++) {
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          B[i * N + j] = A[j * N + i];
        }
      }
    }
    std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Sequential transpose: " << time.count() / NREPET << "s\n";
    std::cout << "Performance: " << (long long) N * N * sizeof(float) / (1e9 * time.count() / NREPET) << "GB/s\n";
    verify(B, N);
  }

  // Transpose the matrix by 8x8 tiles using AVX transpose
  {
    memset(B, 0, N * N * sizeof(float));
    auto start = std::chrono::high_resolution_clock::now();
    for (int repet = 0; repet < NREPET; repet++) {
      for (int i = 0; i < N; i += 8) {
        for (int j = 0; j < N; j += 8) {
          transAVX8x8(&A[i*N + j], &B[j*N + i], tile, N);
        }
      }
    }
    std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
    std::cout << "AVX transpose: " << time.count() / NREPET << "s\n";
    std::cout << "Performance: " << (long long) N * N * sizeof(float) / (1e9 * time.count() / NREPET) << "GB/s\n";
    verify(B, N);
  }

  // Transpose the matrix by 8x8 tiles using AVX transpose and in-place transpose
  {
    memcpy(B, A, N * N * sizeof(float));
    auto start = std::chrono::high_resolution_clock::now();
    for (int repet = 0; repet < NREPET; repet++) {
      for (int i = 0; i < N; i += 8) {
        for (int j = i+8; j < N; j += 8) {
          transAVX8x8InPlace(&A[i*N + j], &A[j*N + i], tile, tile2, N);
        }
        transAVX8x8(&A[i*N + i], &A[i*N + i], tile, N);
      }
    }
    std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
    std::cout << "AVX in-place transpose: " << time.count() / NREPET << "s\n";
    std::cout << "Performance: " << (long long) N * N * sizeof(float) / (1e9 * time.count() / NREPET) << "GB/s\n";
    verify(A, N);
    memcpy(A, B, N * N * sizeof(float));
  }

  // Transpose the matrix by 8x8 tiles using AVX transpose andAVX in-place transpose with OpenMP tasks
  {
    memcpy(B, A, N * N * sizeof(float));
    auto start = std::chrono::high_resolution_clock::now();
    int B1 = 64; // Tile size for each task
    for (int repet = 0; repet < NREPET; repet++) {
      #pragma omp parallel num_threads(16)
      {
        #pragma omp single
        {
          for (int i = 0; i < N; i += B1) {
            for (int j = i; j < N; j += B1) {
              #pragma omp taks firstprivate(i, j, tile, tile2)
              {
                if (i == j) {
                  for (int i1 = i; i1 < i + B1; i1 += 8) {
                    for (int j1 = i1+8; j1 < j + B1; j1 += 8) {
                      transAVX8x8InPlace(&A[i1*N + j1], &A[j1*N + i1], tile, tile2, N);
                    }
                    transAVX8x8(&A[i1*N + i1], &A[i1*N + i1], tile, N);
                  }
                } 
                else {
                  for (int i1 = i; i1 < i + B1; i1 += 8) {
                    for (int j1 = j; j1 < j + B1; j1 += 8) {
                      transAVX8x8InPlace(&A[i1*N + j1], &A[j1*N + i1], tile, tile2, N);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
    std::cout << "AVX in-place transpose with OpenMP tasks: " << time.count() / NREPET << "s\n";
    std::cout << "Performance: " << (long long) N * N * sizeof(float) / (1e9 * time.count() / NREPET) << "GB/s\n";
    verify(A, N);
    memcpy(A, B, N * N * sizeof(float));
  }

  // Free matrices
  _mm_free(A);
  _mm_free(B);

  return 0;
}
