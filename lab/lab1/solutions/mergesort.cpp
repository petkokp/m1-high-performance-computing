// d) The speedup and parallel efficiency are not significant for this example because the size of the array is too small.
// If the size is increased the speedup and the parallel efficiency are improved. (calculated speedup is 0.14 and calculated
// parallel efficiency is 0.035 for this example with 4 threads).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <assert.h>

#define SWAP(a,b) {int tmp = a; a = b; b = tmp;}
#define SIZE 1024
#define NUM_THREADS 4

void verify(int* a, int size);
void merge(int* a, int size, int* temp);
void mergesort(int* a, int size, int* temp);

int main(int argc, char** argv) {
  int* a;
  int* temp;
  int size = SIZE;
  int i;

  //setup
  if (argc >= 2) size = atoi(argv[1]);
  printf("Sorting an array of size %d.\n", size);
  a = (int*)calloc(size, sizeof(int));
  temp = (int*)calloc(size, sizeof(int));
  srand(26);
  for (i = 0; i < size; ++i) {
    a[i] = rand();
  }

  int* a_seq = (int*)calloc(size, sizeof(int));
  memcpy(a_seq, a, size * sizeof(int));

  double start_seq = omp_get_wtime();
  mergesort(a_seq, size, temp);
  double end_seq = omp_get_wtime();
  double time_seq = end_seq - start_seq;
  printf("Sequential execution time: %f seconds\n", time_seq);

  verify(a_seq, size);

  memcpy(a, a_seq, size * sizeof(int));

  double start_par = omp_get_wtime();

  int chunk_size = size / NUM_THREADS;

  #pragma omp parallel
  {
    #pragma omp sections
    {
      #pragma omp section
      {
        mergesort(a, chunk_size, temp);
      }
      #pragma omp section
      {
        mergesort(a + chunk_size, chunk_size, temp + chunk_size);
      }
      #pragma omp section
      {
        mergesort(a + 2 * chunk_size, chunk_size, temp + 2 * chunk_size);
      }
      #pragma omp section
      {
        mergesort(a + 3 * chunk_size, size - 3 * chunk_size, temp + 3 * chunk_size);
      }
    }

    #pragma omp sections
    {
      #pragma omp section
      {
        merge(a, 2 * chunk_size, temp);
      }
      #pragma omp section
      {
        merge(a + 2 * chunk_size, size - 2 * chunk_size, temp + 2 * chunk_size);
      }
    }
  }

  merge(a, size, temp);

  double end_par = omp_get_wtime();
  double time_par = end_par - start_par;
  printf("Parallel execution time: %f seconds\n", time_par);

  verify(a, size);

  double speedup = time_seq / time_par;
  double efficiency = speedup / NUM_THREADS;

  printf("Speedup: %f\n", speedup);
  printf("Parallel efficiency: %f\n", efficiency);

  free(a);
  free(a_seq);
  free(temp);
}

void verify(int* a, int size) {
  int sorted = 1;
  int i;

  for (i = 0; i < size - 1; ++i) sorted &= (a[i] <= a[i + 1]);

  if (sorted) printf("The array was properly sorted.\n");
  else printf("There was an error when sorting the array.\n");
}

void merge(int* a, int size, int* temp) {
  int i1 = 0;
  int i2 = size / 2;
  int it = 0;

  while (i1 < size / 2 && i2 < size) {
    if (a[i1] <= a[i2]) {
      temp[it] = a[i1];
      i1 += 1;
    } else {
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

  memcpy(a, temp, size * sizeof(int));
}

void mergesort(int* a, int size, int* temp) {
  if (size < 2) return;   //nothing to sort
  if (size == 2) {        //only two values to sort
    if (a[0] <= a[1])
      return;
    else {
      SWAP(a[0], a[1]);
      return;
    }
  } else {                //mergesort
    mergesort(a, size / 2, temp);
    mergesort(a + size / 2, size - size / 2, temp + size / 2); //a + size/2: pointer arithmetic
    merge(a, size, temp);
  }
  return;
}
