#include <cstdio>
#include <cstdlib>
#include <omp.h>

void printUsage(int argc, char **argv)
{
  printf("Usage: %s N\n", argv[0]);
  printf("Example: %s 13\n", argv[0]);
}

int fib(int n)
{
  if (n <= 1) {
    return n;
  } else {
    int i, j;
    #pragma omp task shared(i)
    i = fib(n - 1);
    #pragma omp task shared(j)
    j = fib(n - 2);
    #pragma omp taskwait
    return i + j;
  }
}

int main(int argc, char **argv)
{
  // Check the validity of command line arguments and print usage if invalid
  if (argc < 2) {
    printUsage(argc, argv);
    return 0;
  }

  // Read the index of the Fibonacci number to compute
  const int N = atoi(argv[1]);

  // Create threads, then call the function fib(N) from a single thread. fib(N) should create tasks for recursive calls,
  // which will be executed by other available threads.
  #pragma omp parallel
  {
    #pragma omp single
    {
      int result = fib(N);
      printf("Fibonacci(%d) = %d\n", N, result);
    }
  }

  return 0;
}