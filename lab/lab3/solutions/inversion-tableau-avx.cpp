/**
  * Copy an array into another using AVX intrinsics.
  * Compile with flags -O2 -mavx2.
  */

#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

#define NREPET 1024

void showUsage()
{ 
  printf("Usage: ./reverse-array-avx [array-size]\n");
}

int main(int argc, char **argv)
{
  if (argc < 2) { 
    showUsage();
    return 1;
  }
  int dim = std::atoi(argv[1]);
  
  // Allocate and initialize an array of size 'dim' aligned to 32 bytes
  float *array = (float *)_mm_malloc(dim * sizeof(float), 32);
  for (int i = 0; i < dim; i++) { array[i] = i; }
  
  // Reverse the array in place (i.e., without using a second auxiliary array)
  // Repeat NREPET times to better measure execution time
  auto start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    for (int i = 0; i < dim / 2; i++) { 
      float temp = array[i];
      array[i] = array[dim - i];
      array[dim - i] = temp;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> timeSeq = end-start;
  std::cout << std::scientific << "Reversal without AVX: " << timeSeq.count() / NREPET << "s" << std::endl;

  // Reverse the array in place with AVX (i.e., without using a second auxiliary array)
  // Repeat NREPET times to better measure execution time
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    __m256i permIdx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    int i = 0;
    int j = dim - 8;
    for (; i < j; i += 8, j -= 8) {
      __m256 left = _mm256_permutevar8x32_ps(_mm256_load_ps(array + i), permIdx);
      __m256 right = _mm256_permutevar8x32_ps(_mm256_load_ps(array + j), permIdx);
      _mm256_store_ps(array + j, left);
      _mm256_store_ps(array + i, right);
    }
    if (i == j) { // If the array size is not a multiple of 16, reverse the middle in place
      _mm256_store_ps(array + i, _mm256_permutevar8x32_ps(_mm256_load_ps(array + i), permIdx));
    }
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> timeAVX = end-start;
  std::cout << std::scientific << "Reversal with AVX: " << timeAVX.count() / NREPET << "s" << std::endl;

  // Display acceleration and efficiency
  double acceleration = timeSeq.count() / timeAVX.count();
  double efficiency = acceleration / 8;
  std::cout << std::fixed << std::setprecision(2) << "Acceleration: " << acceleration << std::endl;
  std::cout << "Efficiency: " << 100 * efficiency << "%" << std::endl;

  // Deallocate the array
  _mm_free(array);

  return 0;
}
