// b)
// #pragma omp single - block is executed by only one thread, which is not necessarily the master thread
// #pragma omp master - block is only executed by the master thread (thid=0)

// c)
// OMP_NUM_THREADS environment variable - lowest precedence
// omp_set_num_threads() function - overrides the OMP_NUM_THREADS environment variable but is overridden by the num_threads clause
// num_threads clause in the omp parallel construct - highest precedence

#include <stdio.h>
#include <omp.h>

int main() {
    omp_set_num_threads(8);

    #pragma omp parallel default(none) num_threads(6)
    {
        int thid = omp_get_thread_num();
        printf("thid = %d\n", thid);

        int num_threads = omp_get_num_threads();
        printf("total num of threads = %d\n", num_threads);

    #pragma omp single
    {
            int thid = omp_get_thread_num();
            printf("(omp single) Hello World from threadId = %d\n", thid);
    }

        #pragma omp master
    {
        int thid = omp_get_thread_num();;
            printf("(omp master) Hello World from threadId = %d\n", thid);
    }
    }

    return 0;
}