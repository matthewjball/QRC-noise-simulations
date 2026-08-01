#!/bin/bash

gate_vals=(25 35 45 55 65 75 85 95 105 115 125 135 145 155 165 175 185 195 205 215 300 500 700 900)

for gate in "${gate_vals[@]}"
do
python runQRC.py --num_gates=$gate --gate_set=G3 --observables_type=all --err_type=thermal_relaxation --t1=27e-6 --t2=38e-6
done

