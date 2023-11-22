#!/bin/bash
gnuplot << eor
set terminal png 
set output 'output.png' 
set style data linespoints 
set grid 
set title 'Accuracy conversion check'
set datafile separator "," 
set xlabel "epoch" 
set ylabel "accuracy" 
unset label
plot 'data.csv' using 1:2 with lines title "test", 'data.csv' using 1:3 with lines title "train", 'data.csv' using 1:4 with lines title "validate"
eor
