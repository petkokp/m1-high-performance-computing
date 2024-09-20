#include <chrono>
#include <iostream>
#include "omp.h"

inline double f(double x)
{
    return (4 / (1 + x * x));
}

int main()
{
    int i;
    const int N = 100000000;
    double pi = 0.0;
    double s = 1.0 / N;

    // Compute pi sequentially
    auto start = std::chrono::high_resolution_clock::now();

    for (i = 0; i < N; ++i)
    {
        double x_i = i * s;
        double x_ip1 = (i + 1) * s;
        pi += s * (f(x_i) + f(x_ip1)) / 2.0;
    }

    std::cout << "Sequential pi = " << pi << std::endl;
    std::chrono::duration<double> tempsSeq = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Sequential time: " << tempsSeq.count() << "s\n";

    // Compute pi with omp for and reduction
    int numThreads = omp_get_max_threads();

    pi = 0.0;
    omp_set_num_threads(numThreads);

    start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for reduction(+ : pi)
    for (i = 0; i < N; ++i)
    {
        double x_i = i * s;
        double x_ip1 = (i + 1) * s;
        pi += s * (f(x_i) + f(x_ip1)) / 2.0;
    }

    std::cout << "Parallel pi with " << numThreads << " threads = " << pi << std::endl;
    std::chrono::duration<double> tempsOmpFor = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Parallel time with " << numThreads << " threads: " << tempsOmpFor.count() << "s\n";

    // Compute pi with a loop parallalized "by hand"
    pi = 0.0;
    omp_set_num_threads(numThreads);

    start = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        int start_index = (N / num_threads) * thread_id;
        int end_index = (N / num_threads) * (thread_id + 1);

        if (thread_id == num_threads - 1)
        {
            end_index = N;
        }

        double local_pi = 0.0;

        for (int i = start_index; i < end_index; ++i)
        {
            double x_i = i * s;
            double x_ip1 = (i + 1) * s;
            local_pi += s * (f(x_i) + f(x_ip1)) / 2.0;
        }

#pragma omp atomic
        pi += local_pi;
    }

    std::cout << "Parallel pi (by hand) with " << numThreads << " threads = " << pi << std::endl;
    std::chrono::duration<double> tempsByHand = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Parallel time (by hand) with " << numThreads << " threads: " << tempsByHand.count() << "s\n";

    return 0;
}
