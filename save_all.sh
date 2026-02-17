#!/usr/bin/sh

# two loops, instead mv runs quicker than pdf file can generate and it is easy to break

mkdir -p simulation_plots
for num in $(seq 50); do
  python read_simulations.py $num
done
sleep 2s
for num in $(seq 50); do
  mv simulation_$num.pdf simulation_plots/
done
