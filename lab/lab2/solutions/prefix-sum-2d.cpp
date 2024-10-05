#include <iostream>
#include <chrono>
#include "omp.h"

#define N 16
#define K 4
#define NTASKS (N / K)

double A[N][N];
double B[N][N];
bool deps[NTASKS + 1][NTASKS + 1];

void printArray(double tab[N][N])
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%4.0lf ", tab[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char **argv)
{
  using clock = std::chrono::high_resolution_clock;
  using duration = std::chrono::duration<double>;

  std::chrono::time_point<clock> start, end;

  // Version 1: Using omp for loop
  start = clock::now();
#pragma omp parallel default(none) shared(A)
  {
#pragma omp for collapse(2)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        A[i][j] = i + j;
  }
  end = clock::now();
  std::cout << "Time for version 1 (omp for): " 
            << std::chrono::duration_cast<duration>(end - start).count() 
            << " seconds\n";

  if (N < 20) {
    printf("Array A after version 1 (omp for):\n");
    printArray(A);
  }

  // Version 2: Using tasks (block-based)
  start = clock::now();
#pragma omp parallel default(none) shared(A)
  {
#pragma omp for collapse(2)
    for (int ti = 0; ti < NTASKS; ti++) {
      for (int tj = 0; tj < NTASKS; tj++) {
#pragma omp task default(none) firstprivate(ti, tj) shared(A)
        for (int i = ti * K; i < (ti + 1) * K; i++) {
          for (int j = tj * K; j < (tj + 1) * K; j++) {
            A[i][j] = i + j;
          }
        }
      }
    }
  }
  end = clock::now();
  std::cout << "Time for version 2 (tasks): " 
            << std::chrono::duration_cast<duration>(end - start).count() 
            << " seconds\n";

  if (N < 20) {
    printf("Array A after version 2 (tasks):\n");
    printArray(A);
  }

  // Non-blocked version for computing B
  start = clock::now();
#pragma omp parallel default(none) shared(A, B, deps)
 {
#pragma omp single
   {
     for (int i = 0; i < N; i++) {
       for (int j = 0; j < N; j++) {
#pragma omp task default(none) firstprivate(i, j) shared(A, B) \
         depend(in:A[i - 1][j - 1],A[i - 1][j],A[i][j - 1]) depend(out:A[i][j])
         {
           if (i == 0) { 
             if (j == 0) {
               B[i][j] = A[i][j];
             } else {
               B[i][j] = B[i][j - 1] + A[i][j];
             }
           } else if (j == 0) {
             B[i][j] = B[i - 1][j] + A[i][j];
           } else {
             B[i][j] = B[i - 1][j] + B[i][j - 1] - B[i - 1][j - 1] + A[i][j];
           }
         }
       }
     }
   }
 }
  end = clock::now();
  std::cout << "Time for non-blocked version: " 
            << std::chrono::duration_cast<duration>(end - start).count() 
            << " seconds\n";

  if (N < 20) {
    printf("Non-blocked version - array B:\n");
    printArray(B);
  }

  // Blocked version for computing B
  start = clock::now();
#pragma omp parallel default(none) shared(A, B, deps)
  {
#pragma omp single
    {
      for (int ti = 0; ti < NTASKS; ti++) {
        for (int tj = 0; tj < NTASKS; tj++) {
#pragma omp task default(none) firstprivate(ti, tj) shared(A, B) \
          depend(out:deps[ti + 1][tj + 1]) depend(in:deps[ti][tj],deps[ti + 1][tj],deps[ti][tj + 1])
          for (int i = ti * K; i < (ti + 1) * K; i++) {
            for (int j = tj * K; j < (tj + 1) * K; j++) {
              if (i == 0) { 
                if (j == 0) {
                  B[i][j] = A[i][j];
                } else {
                  B[i][j] = B[i][j - 1] + A[i][j];
                }
              } else if (j == 0) {
                B[i][j] = B[i - 1][j] + A[i][j];
              } else {
                B[i][j] = B[i - 1][j] + B[i][j - 1] - B[i - 1][j - 1] + A[i][j];
              }
            }
          }
        }
      }
    }
  }
  end = clock::now();
  std::cout << "Time for blocked version: " 
            << std::chrono::duration_cast<duration>(end - start).count() 
            << " seconds\n";

  if (N < 20) {
    printf("Blocked version - array B:\n");
    printArray(B);
  }

  return 0;
}
