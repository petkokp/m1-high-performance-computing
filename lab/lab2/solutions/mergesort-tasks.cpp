#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <chrono>
#include "omp.h"

#define SWAP(a, b) { auto tmp = a; a = b; b = tmp; }
#define SIZE 1024
#define K 100000 // the minimum size threshold for task creation

/**
 * Verify if the given array is sorted
 * Verifier si le tableau donne est trie
 */
bool isSorted(int *a, int size)
{
  bool sorted = true;
#pragma omp parallel default(none) shared(a, size, sorted)
  {
#pragma omp for reduction(&:sorted)
    for (int i = 0; i < size - 1; i++) sorted &= (a[i] <= a[i + 1]);
  }

  return sorted;
}

/**
 * Merge two sorted subarrays a[0...size/2 - 1] and a[size/2...size - 1] into a single sorted array a[0...size - 1].
 * Use temp as a temporary buffer for merge
 */
void merge(int* a, int size, int* temp)
{
  int i1 = 0;
  int i2 = size / 2;
  int it = 0;

  // Perform the merge on temp
  while(i1 < size / 2 && i2 < size) {
    if (a[i1] <= a[i2]) {
      temp[it] = a[i1];
      i1 += 1;
    }
    else {
      temp[it] = a[i2];
      i2 += 1;
    }
    it += 1;
  }

  while (i1 < size / 2) {
    temp[it] = a[i1];
    i1++;
    it++;
  }
  while (i2 < size) {
    temp[it] = a[i2];
    i2++;
    it++;
  }

  // Copy temp back into a
  memcpy(a, temp, size * sizeof(int));
}

/**
 * Sort the array a with size elements, using temp as a temporary merge buffer
 */
void mergesort(int *a, int size, int *temp)
{
  if (size < 2) return;  // Nothing to sort
  if (size == 2) {       // Only two elements to sort
    if (a[0] > a[1]) SWAP(a[0], a[1]);
    return;
  }

  if (size > K) {
    #pragma omp task shared(a, temp) firstprivate(size)
    {
      mergesort(a, size / 2, temp);
    }

    #pragma omp task shared(a, temp) firstprivate(size)
    {
      mergesort(a + size / 2, size - size / 2, temp + size / 2);
    }

    // Wait for both tasks to complete
    #pragma omp taskwait
  } else {
    mergesort(a, size / 2, temp);
    mergesort(a + size / 2, size - size / 2, temp + size / 2);
  }

  // Merge the sorted halves
  merge(a, size, temp);
}

void printUsage(int argc, char **argv)
{
  printf("Usage: %s [size]\n", argv[0]);
  printf("Example: %s 1024\n", argv[0]);
}

int main(int argc, char **argv)
{
  int *a;
  int *temp;
  int size;

  // Read array size and initialize
  if (argc < 2) { 
    printUsage(argc, argv);
    return 0;
  }
  size = atoi(argv[1]);
  a = (int *) malloc(size * sizeof(int));
  temp = (int *) malloc(size * sizeof(int));
  srand(0);

#pragma omp parallel default(none) shared(a, size)
  {
#pragma omp for
    for (int i = 0; i < size; ++i) {
      a[i] = size - i;
    }
  }

  // Sort the array a[size] using mergesort in parallel, using temp as a temporary buffer
  // You need to create threads, call mergesort from one thread, then create tasks for recursive calls.
  auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel default(none) shared(a, temp, size)
  {
    #pragma omp single
    {
      mergesort(a, size, temp);
    }
  }

  std::chrono::duration<double> sortTime = std::chrono::high_resolution_clock::now() - start;
  printf("Sorting took %.4lf seconds.\n", sortTime.count());

  // Verify if the array is sorted
  if (isSorted(a, size)) { printf("The array was properly sorted.\n"); }
  else { printf("There was an error when sorting the array.\n"); }

  // Deallocate the arrays
  free(a);
  free(temp);
  
  return 0;
}
