#include <iostream>
#include <chrono>
#include <vector>
#include "omp.h"

#define NREPET 1024
#define SMALL_MATRIX_THRESHOLD 64 // threshold to switch between sequential and parallel

int main(int argc, char **argv)
{
    std::cout << "Matrix-vector product with OpenMP\n";
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [num-rows / columns]\n";
        std::cout << "  Example: " << argv[0] << " 1024\n";
        return 1;
    }
    int dim = std::atoi(argv[1]); 

    std::vector<double> A(dim * dim);
    std::vector<double> x(dim);
    std::vector<double> b(dim, 0);

    // Initialize A and x so that A(i, j) = i + j and x(j) = 1.
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            A[i * dim + j] = i + j;
        }
        x[i] = 1;
    }

    // Sequential execution: b = A * x, repeated NREPET times
    auto start = std::chrono::high_resolution_clock::now();
    for (int repet = 0; repet < NREPET; repet++) {
        for (int i = 0; i < dim; i++) {
            b[i] = 0;
            for (int j = 0; j < dim; j++) {
                b[i] += A[i * dim + j] * x[j];
            }
        }
    }
    std::chrono::duration<double> seqTime = std::chrono::high_resolution_clock::now() - start;
    std::cout << std::scientific << "Sequential execution time: " << seqTime.count() / NREPET << "s" << std::endl;

    // Parallel execution using omp for: b = A * x, repeated NREPET times
    start = std::chrono::high_resolution_clock::now();
    for (int repet = 0; repet < NREPET; repet++) {
        #pragma omp parallel for
        for (int i = 0; i < dim; i++) {
            b[i] = 0;
            for (int j = 0; j < dim; j++) {
                b[i] += A[i * dim + j] * x[j];
            }
        }
    }
    std::chrono::duration<double> parTime = std::chrono::high_resolution_clock::now() - start;
    std::cout << std::scientific << "Parallel execution time with omp for: " << parTime.count() / NREPET << "s" << std::endl;

    // Speedup and Efficiency calculation
    double speedup = seqTime.count() / parTime.count();
    int numThreads = omp_get_max_threads();
    double efficiency = speedup / numThreads;

    std::cout << "Speedup: " << speedup << std::endl;
    std::cout << "Efficiency: " << efficiency << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int repet = 0; repet < NREPET; repet++) {
        if (dim > SMALL_MATRIX_THRESHOLD) {
            #pragma omp parallel for
            for (int i = 0; i < dim; i++) {
                b[i] = 0;
                for (int j = 0; j < dim; j++) {
                    b[i] += A[i * dim + j] * x[j];
                }
            }
        } else {
            for (int i = 0; i < dim; i++) {
                b[i] = 0;
                for (int j = 0; j < dim; j++) {
                    b[i] += A[i * dim + j] * x[j];
                }
            }
        }
    }
    std::chrono::duration<double> conditionalParTime = std::chrono::high_resolution_clock::now() - start;
    std::cout << std::scientific << "Parallel execution with conditional omp for: " << conditionalParTime.count() / NREPET << "s" << std::endl;

    // Check the result. b(i) is expected to be (dim - 1) * dim / 2 + i * dim
    for (int i = 0; i < dim; i++) {
        double expected = (dim - 1) * (double)dim / 2.0 + (double)i * dim;
        if (b[i] != expected) {
            std::cout << "Incorrect value: b[" << i << "] = " << b[i] << " != " << expected << std::endl;
            break;
        }
        if (i == dim - 1) {
            std::cout << "Matrix-vector multiplication succeeded!\n";
        }
    }

    return 0;
}
