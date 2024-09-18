#!/bin/bash

# Number of maximum threads (adjust to your CPU)
MAX_THREADS=24

# Output file to store the results
OUTPUT_FILE="execution_times.txt"

# Clean the output file
echo "Threads, Execution time (s)" > $OUTPUT_FILE

# Compile the program (make sure OpenMP is enabled)
g++ -fopenmp -o sum-array-openmp sum-array-openmp.cpp

# Run the program for 1 to MAX_THREADS threads
for ((threads=1; threads<=MAX_THREADS; threads++)); do
    echo "Running with $threads thread(s)..."
    
    # Run the program and capture the execution time
    EXEC_TIME=$(./sum-array-openmp $threads | grep "Execution time" | awk '{print $3}')

    echo $EXEC_TIME
    
    # Output the number of threads and execution time to the output file
    echo "$threads, $EXEC_TIME" >> $OUTPUT_FILE
done

echo "All tests completed! Results stored in $OUTPUT_FILE."
