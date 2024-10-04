#include <iostream>
#include <chrono>
#include <cmath>
#include "omp.h"

#define N 8192

// a) The default scheduling strategy in OpenMP is `static`, and the default chunk size is (number of iterations) / (number of threads).
// b) Added parallelization with static scheduling and a chunk size of 256.
// c) The default chunk size for dynamic scheduling is 1.
// d) Added parallelization with dynamic scheduling and a chunk size of 256.
// e) The default chunk size for guided scheduling varies dynamically, starting with larger chunks and reducing the chunk size over time.
// f) Added parallelization with guided scheduling and a chunk size of 256.
// g) The best results will depend on the computational cost distribution. Since Goldbach's algorithm grows quadratically, guided scheduling may balance load better than static or dynamic scheduling.

/**
  * Check if x is a prime number
  */
bool isPrime(int x) {
    if (x < 2) { return false; }
    for (int i = 2; i <= x / 2; i++) {
        if (x % i == 0) { return false; }
    }
    return true;
}

/**
  * Count the number of pairs (i, j) such that i + j = x and both i and j are prime numbers.
  */
int goldbach(int x) {
    int count = 0;
    if (x <= 2) { return 0; }
    for (int i = 2; i <= x / 2; i++) {
        if (isPrime(i) && isPrime(x - i)) { count++; }
    }
    return count;
}

int main() {
    int goldbachTrue;
    int numPairs[N];
    for (int i = 0; i < N; i++) { numPairs[i] = 0; }

    // Sequential version
    auto start = std::chrono::high_resolution_clock::now();
    goldbachTrue = 1;
    for (int i = 4; i < N; i += 2) {
        numPairs[i] = goldbach(i);
        if (numPairs[i] == 0) { goldbachTrue = 0; }
    }
    if (goldbachTrue) { std::cout << "Goldbach's conjecture is true" << std::endl; }
    else { std::cout << "Goldbach's conjecture is false" << std::endl; }
    std::chrono::duration<double> seqTime = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Sequential time: " << seqTime.count() << "s\n";

    // Parallelization without specifying scheduling (default scheduling)
    start = std::chrono::high_resolution_clock::now();
    goldbachTrue = 1;
    #pragma omp parallel for reduction(&:goldbachTrue)
    for (int i = 4; i < N; i += 2) {
        numPairs[i] = goldbach(i);
        if (numPairs[i] == 0) { goldbachTrue = 0; }
    }
    if (goldbachTrue) { std::cout << "Goldbach's conjecture is true" << std::endl; }
    else { std::cout << "Goldbach's conjecture is false" << std::endl; }
    std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Time with default scheduling: " << time.count() << "s\n";

    // Parallelization with static scheduling and chunk size 256
    start = std::chrono::high_resolution_clock::now();
    goldbachTrue = 1;
    #pragma omp parallel for schedule(static, 256) reduction(&:goldbachTrue)
    for (int i = 4; i < N; i += 2) {
        numPairs[i] = goldbach(i);
        if (numPairs[i] == 0) { goldbachTrue = 0; }
    }
    if (goldbachTrue) { std::cout << "Goldbach's conjecture is true" << std::endl; }
    else { std::cout << "Goldbach's conjecture is false" << std::endl; }
    time = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Time with schedule(static, 256): " << time.count() << "s\n";

    // Parallelization with dynamic scheduling
    start = std::chrono::high_resolution_clock::now();
    goldbachTrue = 1;
    #pragma omp parallel for schedule(dynamic) reduction(&:goldbachTrue)
    for (int i = 4; i < N; i += 2) {
        numPairs[i] = goldbach(i);
        if (numPairs[i] == 0) { goldbachTrue = 0; }
    }
    if (goldbachTrue) { std::cout << "Goldbach's conjecture is true" << std::endl; }
    else { std::cout << "Goldbach's conjecture is false" << std::endl; }
    time = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Time with schedule(dynamic): " << time.count() << "s\n";

    // Parallelization with dynamic scheduling and chunk size 256
    start = std::chrono::high_resolution_clock::now();
    goldbachTrue = 1;
    #pragma omp parallel for schedule(dynamic, 256) reduction(&:goldbachTrue)
    for (int i = 4; i < N; i += 2) {
        numPairs[i] = goldbach(i);
        if (numPairs[i] == 0) { goldbachTrue = 0; }
    }
    if (goldbachTrue) { std::cout << "Goldbach's conjecture is true" << std::endl; }
    else { std::cout << "Goldbach's conjecture is false" << std::endl; }
    time = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Time with schedule(dynamic, 256): " << time.count() << "s\n";

    // Parallelization with guided scheduling
    start = std::chrono::high_resolution_clock::now();
    goldbachTrue = 1;
    #pragma omp parallel for schedule(guided) reduction(&:goldbachTrue)
    for (int i = 4; i < N; i += 2) {
        numPairs[i] = goldbach(i);
        if (numPairs[i] == 0) { goldbachTrue = 0; }
    }
    if (goldbachTrue) { std::cout << "Goldbach's conjecture is true" << std::endl; }
    else { std::cout << "Goldbach's conjecture is false" << std::endl; }
    time = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Time with schedule(guided): " << time.count() << "s\n";

    // Parallelization with guided scheduling and chunk size 256
    start = std::chrono::high_resolution_clock::now();
    goldbachTrue = 1;
    #pragma omp parallel for schedule(guided, 256) reduction(&:goldbachTrue)
    for (int i = 4; i < N; i += 2) {
        numPairs[i] = goldbach(i);
        if (numPairs[i] == 0) { goldbachTrue = 0; }
    }
    if (goldbachTrue) { std::cout << "Goldbach's conjecture is true" << std::endl; }
    else { std::cout << "Goldbach's conjecture is false" << std::endl; }
    time = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Time with schedule(guided, 256): " << time.count() << "s\n";

    return 0;
}
