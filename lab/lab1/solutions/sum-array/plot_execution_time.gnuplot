set terminal pngcairo size 800,600 enhanced font 'Verdana,12'
set output 'execution_times.png'

set title "Execution time vs threads"
set xlabel "Number of threads"
set ylabel "Execution time (s)"
set grid

set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5  # Blue line

plot 'execution_times.txt' using 1:2 with linespoints ls 1 title 'Execution time'
