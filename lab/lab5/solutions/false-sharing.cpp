//            1 thread	2 threads 4 threads 8 threads
// 1 stride	  5.33587s	5.3055s	  4.65022s	4.37799s
// 2 strides	3.15605s	3.67531s	3.05273s	2.9497s
// 4 strides	1.84341s	1.72321s	1.94426s	1.79241s
// 8 strides	0.919007s	0.912719s	1.22285s	0.936331s
// 16 strides	0.477303s	0.469474s	0.521821s	0.465284s
// 32 strides	0.47822s	0.48302s	0.483496s	0.529152s
// 64 strides	0.649425s	0.605507s	0.483519s	0.56552s

// b) In sequential execution, increasing the STRIDE improves performance at first
// but only up to a point. When the stride gets larger, the performance stops getting
// better because the data accesses are spread too far apart, causing the cache to be
// less efficient. So, while small strides can help, going too big actually slows things down.

// c) Oarallel execution gets faster as you increase the STRIDE, but only up to a point.
// Small strides cause false sharing and bigger strides fix this by keeping threads away
// from each other's data. But after a certain stride size, performance stops improving because
// the memory accesses are too spread out to gain any more benefits.

// d) The effect is lost around 32 or 64 strides. At this point, false sharing is already minimized,
// so increasing the stride further doesn't help. Instead, it just spreads the memory accesses too
// far apart, reducing cache efficiency because threads are no longer benefiting from accessing
// nearbydata in the same cache line.

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include "omp.h"

#define NREPEAT 128
#define NTHREADMAX 8
#define STRIDE 64

int main()
{
  int N = 10000000;
  float sum[NTHREADMAX * STRIDE] __attribute__((aligned(64))) = {0};
  std::vector<float> vec(N);
  vec[0] = 0;
  for (int i = 1; i < N; i++) { 
    vec[i] = 1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int repeat = 0; repeat < NREPEAT; repeat++) {
    for (int i = 0; i < NTHREADMAX; i++) { sum[i * STRIDE] = 0.0; }
#pragma omp parallel
    {
      int thid = omp_get_thread_num();
#pragma omp for
      for (int i = 0; i < N; i++) {
        sum[thid * STRIDE] += vec[i];
      }
    }
  }
  std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
  std::cout << "Time: " << time.count() << "s\n";

  float sumFinal = 0.0;
  for (int i = 0; i < NTHREADMAX; i++) { sumFinal += sum[i]; }
  printf("sum = %f", sumFinal);

  return 0;
}
