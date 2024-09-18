// d) shouldn't use nowait because there is a dependency between the initialization and summation loops

// f)
// Threads, Execution time (s)
// 1, 0.119062s
// 2, 0.118971s
// 3, 0.110698s
// 4, 0.115723s
// 5, 0.141578s
// 6, 0.120345s
// 7, 0.119687s
// 8, 0.11012s
// 9, 0.112636s
// 10, 0.132172s
// 11, 0.109832s
// 12, 0.0715165s
// 13, 0.0615115s
// 14, 0.0676461s
// 15, 0.0739857s
// 16, 0.0676885s
// 17, 0.0747464s
// 18, 0.0676339s
// 19, 0.0601174s
// 20, 0.067783s
// 21, 0.06195s
// 22, 0.0683312s
// 23, 0.0584015s
// 24, 0.0615228s

#include <cstdio>
#include <vector>
#include "unistd.h"
#include "omp.h"
#include <iostream>
#include <chrono>

#define MAX_THREADS 1000

int main(int argc, char *argv[])
{
    if (argc == 2)
    {
        int num_threads = std::stoi(argv[1]);
        omp_set_num_threads(num_threads);
        printf("Using %d threads\n", num_threads);
    }

    int i;
    int N = 10000000;
    std::vector<double> A(N);
    double sum = 0.0;
    std::vector<double> localSum(MAX_THREADS, 0.0);

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel default(none) num_threads(4) shared(A, localSum, sum, N)
    {
        int thid = omp_get_thread_num();

#pragma omp for // nowait

        for (int i = 0; i < N; i++)
        {
            A[i] = i;
        }

#pragma omp sections
        {
#pragma omp section
            {
                int thid = omp_get_thread_num();

                for (int i = 0; i < N / 4; i++)
                {
                    localSum[thid] = localSum[thid] + A[i];
                }
            }
#pragma omp section
            {
                int thid = omp_get_thread_num();

                for (int i = N / 4; i < N / 2; i++)
                {
                    localSum[thid] = localSum[thid] + A[i];
                }
            }

#pragma omp section
            {
                int thid = omp_get_thread_num();

                for (int i = N / 2; i < 3 * N / 4; i++)
                {
                    localSum[thid] = localSum[thid] + A[i];
                }
            }
#pragma omp section
            {
                int thid = omp_get_thread_num();

                for (int i = 3 * N / 4; i < N; i++)
                {
                    localSum[thid] = localSum[thid] + A[i];
                }
            }
        }

#pragma omp atomic
        sum = sum + localSum[thid];
    }

    std::cout << "Sum is " << sum << std::endl;
    std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Execution time: " << time.count() << "s\n";

    return 0;
}