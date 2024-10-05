/**
  * Copying one array into another using AVX intrinsics.
  * Compile with the flags -O2 -mavx2.
  */

// e) The unrolled version is faster for 1000 subsequent executions for N=1024 but slower if the number of executions is increased much more

#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

#define NREPET 1024

void displayUsage()
{ 
  printf("Usage: ./copy-vector-avx [array-size]\n");
}

int main(int argc, char **argv)
{
  if (argc < 2) { 
    displayUsage();
    return 1;
  }
  int dim = std::atoi(argv[1]);
  
  // Allocate and initialize two arrays of floats of size dim aligned by 32 bytes
  float* tab0 = (float*)_mm_malloc(dim * sizeof(float), 32);
  float* tab1 = (float*)_mm_malloc(dim * sizeof(float), 32);
  for (int i = 0; i < dim; i++) {
    tab0[i] = i;
    tab1[i] = 0;
  }
  
  // Copy tab0 to tab1 in a scalar way (non-vectorized code).
  // Repeat NREPET times to better measure execution time
  auto start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    for(int i = 0; i < dim; i++){
      tab1[i] = tab0[i];
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diffSeq = end - start;
  std::cout << std::scientific << "Copy without AVX: " << diffSeq.count() << "s" << std::endl;

  // Copy tab0 to tab1 in a vectorized way using AVX
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    int i;
    for(i = 0; i < dim - 7; i += 8) {
      __m256 r0 = _mm256_load_ps(tab0 + i);
      _mm256_store_ps(tab1 + i, r0);
    }
    for (; i < dim; i++) { tab1[i] = tab0[i]; }
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diffPar = end-start;
  std::cout << std::scientific << "Copy with AVX: " << diffPar.count() << "s" << std::endl;

  // Display speedup and efficiency
  double speedup = diffSeq.count() / diffPar.count();
  double efficiency = speedup / 8;
  std::cout << std::fixed << std::setprecision(2) << "Speedup: " << speedup << std::endl;
  std::cout << "Efficiency: " << 100 * efficiency << "%" << std::endl;

  // Copy tab0 to tab1 in a vectorized way using AVX with loop unrolling factor 4
  start = std::chrono::high_resolution_clock::now();
  for (int repet = 0; repet < NREPET; repet++) {
    int i;
    for(i = 0; i < dim - 31; i += 32) {
      __m256 r0 = _mm256_load_ps(tab0 + i);
      __m256 r1 = _mm256_load_ps(tab0 + i + 8);
      __m256 r2 = _mm256_load_ps(tab0 + i + 16);
      __m256 r3 = _mm256_load_ps(tab0 + i + 24);
      _mm256_store_ps(tab1 + i, r0);
      _mm256_store_ps(tab1 + i + 8, r1);
      _mm256_store_ps(tab1 + i + 16, r2);
      _mm256_store_ps(tab1 + i + 24, r3);
    }
    for (; i < dim; i++) { tab1[i] = tab0[i]; }
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diffParUnrolled = end-start;
  std::cout << std::scientific << "Copy with AVX and unrolling: " << diffParUnrolled.count() << "s" << std::endl;
 
  // Display speedup and efficiency
  speedup = diffSeq.count() / diffParUnrolled.count();
  efficiency = speedup / 8;
  std::cout << std::fixed << std::setprecision(2) << "Speedup: " << speedup << std::endl;
  std::cout << "Efficiency: " << 100 * efficiency << "%" << std::endl;

  // Deallocate the arrays tab0 and tab1
  _mm_free(tab0);
  _mm_free(tab1);

  return 0;
}
