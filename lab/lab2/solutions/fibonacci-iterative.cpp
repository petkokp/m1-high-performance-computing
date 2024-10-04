#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include "omp.h"

void printUsage(int argc, char **argv) {
    printf("Usage: %s N\n", argv[0]);
    printf("Example: %s 13\n", argv[0]);
}

int main(int argc, char **argv) {
    // Check the validity of command line arguments and print usage if invalid
    if (argc < 2) { 
        printUsage(argc, argv);
        return 0;
    }

    // Read the index of the Fibonacci number to compute
    const int N = atoi(argv[1]);

    // Allocate and initialize the array containing Fibonacci numbers
    int fib[N];
    fib[0] = 0;
    fib[1] = 1;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 2; i < N; i++) {
                #pragma omp task depend(in: fib[i - 1], fib[i - 2]) depend(out: fib[i])
                {
                    fib[i] = fib[i - 1] + fib[i - 2];
                }
            }
        }

        #pragma omp taskwait
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Print all computed Fibonacci numbers until N
    printf("Fibonacci numbers: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", fib[i]);
    }
    printf("\n");

    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    return 0;
}
