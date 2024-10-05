#include <iostream>
#include <vector>
#include "immintrin.h"
#include <chrono>

void printUsage(int argc, char **argv)
{
  printf("Usage: %s N\n", argv[0]);
  printf("Example: %s 1024\n", argv[0]);
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    printUsage(argc, argv);
    return 0;
  }
  int N = std::atoi(argv[1]);

  // Allocate and initialize the matrix A and vectors x, b
  std::vector<float> A(N * N);
  std::vector<float> x(N);
  std::vector<float> b(N);
  for (int i = 0; i < N; i++) {
    x[i] = 1;
    for (int j = 0; j < N; j++) {
      A[i * N + j] = i + j;
    }
  }

  // Scalar matrix-vector multiplication code, for reference
  // for (int i = 0; i < N; i++) {
  //   for (int j = 0; j < N; j++) {
  //     b[i] = b[i] + A[i * N + j] * x[j];
  //   }
  // }

  // Vectorized matrix-vector multiplication
  float bvecarr[8];
  __m256 bvec;
  __m256 xvec;
  __m256 avec;
  for (int i = 0; i < N; i++) {
    bvec = _mm256_set1_ps(0.0f);
    for (int j = 0; j < N; j += 8) {
      avec = _mm256_loadu_ps(&A[i * N + j]);
      xvec = _mm256_loadu_ps(&x[j]);
      bvec = _mm256_fmadd_ps(avec, xvec, bvec);
    }
    _mm256_storeu_ps(&bvecarr[0], bvec);
    bvecarr[0] += bvecarr[4];
    bvecarr[1] += bvecarr[5];
    bvecarr[2] += bvecarr[6];
    bvecarr[3] += bvecarr[7];
    bvecarr[0] += bvecarr[2];
    bvecarr[1] += bvecarr[3];
    b[i] = bvecarr[0] + bvecarr[1];
  }

  // Verify the results
  int correct = 1;
  for (int i = 0; i < N; i++) {
    if (b[i] !=  (N + i - 1) * (N + i) / 2.0 - i * (i - 1) / 2) {
      printf("b[%d] = %f is incorrect; b[%d] should be %f\n", i, b[i], i, (N+i-1)*(N+i)/2.0);
      correct = 0;
      break;
    }
  }
  if (correct) {
    printf("The result is correct!\n");
  } else {
    printf("The result is not correct!\n");
  }

  return 0;
}
