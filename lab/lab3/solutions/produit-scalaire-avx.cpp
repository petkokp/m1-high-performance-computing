/**
  * Dot product of two arrays using AVX and FMA intrinsics.
  * Compile with the flags -O2 -mavx2 -mfma
  */

// e) multiplications ad additions are faster in some cases with fused-multiply-add (FMA) operations

#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

#define NREPET 1024

inline float dotProduct(float *A, float *B, int size)
{
  float result = 0.0f;
  for (size_t i = 0; i < size; i++) { 
    result += A[i] * B[i];
  }
  return result;
}

inline float dotProductAVX(float *A, float *B, int size)
{
  __m256 result = _mm256_set1_ps(0.0f);
  for (size_t i = 0; i < size; i += 8) {
    __m256 vecA = _mm256_load_ps(A + i);
    __m256 vecB = _mm256_load_ps(B + i);
    __m256 product = _mm256_mul_ps(vecA, vecB);
    result = _mm256_add_ps(result, product);
  }
  float resultArray[8] __attribute__((aligned(32)));
  _mm256_store_ps(resultArray, result);
  return resultArray[0] + resultArray[1] + resultArray[2] + resultArray[3] + resultArray[4] + resultArray[5] + resultArray[6] + resultArray[7];
}

inline float dotProductAVXFMA(float *A, float *B, int size)
{
  __m256 result = _mm256_set1_ps(0.0f);
  for (size_t i = 0; i < size; i += 8) {
    __m256 vecA = _mm256_load_ps(A + i);
    __m256 vecB = _mm256_load_ps(B + i);
    result = _mm256_fmadd_ps(vecA, vecB, result);
  }
  float resultArray[8] __attribute__((aligned(32)));
  _mm256_store_ps(resultArray, result);
  return resultArray[0] + resultArray[1] + resultArray[2] + resultArray[3] + resultArray[4] + resultArray[5] + resultArray[6] + resultArray[7];
}

inline float dotProductAVXFMALoopUnrolling(float *__restrict__ A, float *__restrict__ B, int size)
{
  __m256 result1 = _mm256_set1_ps(0.0f);
  __m256 result2 = _mm256_set1_ps(0.0f);
  __m256 result3 = _mm256_set1_ps(0.0f);
  __m256 result4 = _mm256_set1_ps(0.0f);
  for (size_t i = 0; i < size; i += 32) {
    __m256 vecA1 = _mm256_load_ps(A + i);
    __m256 vecA2 = _mm256_load_ps(A + i + 8);
    __m256 vecA3 = _mm256_load_ps(A + i + 16);
    __m256 vecA4 = _mm256_load_ps(A + i + 24);
    __m256 vecB1 = _mm256_load_ps(B + i);
    __m256 vecB2 = _mm256_load_ps(B + i + 8);
    __m256 vecB3 = _mm256_load_ps(B + i + 16);
    __m256 vecB4 = _mm256_load_ps(B + i + 24);
    result1 = _mm256_fmadd_ps(vecA1, vecB1, result1);
    result2 = _mm256_fmadd_ps(vecA2, vecB2, result2);
    result3 = _mm256_fmadd_ps(vecA3, vecB3, result3);
    result4 = _mm256_fmadd_ps(vecA4, vecB4, result4);
  }
  result1 = _mm256_add_ps(result1, result2);
  result3 = _mm256_add_ps(result3, result4);
  result1 = _mm256_add_ps(result1, result3);
  float resultArray[8] __attribute__((aligned(32)));
  _mm256_store_ps(resultArray, result1);
  return resultArray[0] + resultArray[1] + resultArray[2] + resultArray[3] + resultArray[4] + resultArray[5] + resultArray[6] + resultArray[7];
}

int main(int argc, char **argv)
{
  if (argc < 2) {
    std::cout << "Usage: \n  " << argv[0] << " [array-size]\n";
    return 1;
  }
  int dim = std::atoi(argv[1]);
  if (dim % 8) {
    std::cout << "The array size must be a multiple of 8.\n";
    return 1;
  }

  float* array0;
  float* array1;
  
  // Allocate and initialize two floating-point arrays of size 'dim', aligned to 32 bytes
  array0 = (float*) _mm_malloc(dim * sizeof(float), 32);
  array1 = (float*) _mm_malloc(dim * sizeof(float), 32);
  for (int i = 0; i < dim; i++) { 
    array0[i] = i;
    array1[i] = i;
  }
  
  // Non-vectorized dot product. Repeated NREPET times to better measure execution time
  auto start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    dotProduct(array0, array1, dim);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> timeSeq = end-start;
  std::cout << std::scientific << "Dot product without AVX: " << timeSeq.count() / NREPET << "s" << std::endl;

  // Vectorized dot product with AVX.
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    dotProductAVX(array0, array1, dim);
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> timeAVX = end-start;
  std::cout << std::scientific << "Dot product with AVX: " << timeAVX.count() / NREPET << "s" << std::endl;

  // Vectorized dot product with AVX FMA.
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    dotProductAVXFMA(array0, array1, dim);
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> timeAVXFMA = end-start;
  std::cout << std::scientific << "Dot product with AVX FMA: " << timeAVXFMA.count() / NREPET << "s" <<
    std::endl;

  // Vectorized dot product with AVX FMA and loop unrolling.
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    dotProductAVXFMALoopUnrolling(array0, array1, dim);
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> timeAVXFMALoopUnrolled = end-start;
  std::cout << std::scientific << "Dot product with AVX FMA loop unrolling: " << timeAVXFMALoopUnrolled.count() /
    NREPET << "s" << std::endl;

  // Display acceleration and efficiency
  double acceleration = timeSeq.count() / timeAVXFMALoopUnrolled.count();
  double efficiency = acceleration / 8;
  std::cout << "Acceleration: " << std::setprecision(2) << std::fixed << acceleration << std::endl;
  std::cout << "Efficiency: " << std::setprecision(2) << std::fixed << efficiency << std::endl;

  // Deallocate the arrays array0 and array1
  _mm_free(array0);
  _mm_free(array1);

  return 0;
}
