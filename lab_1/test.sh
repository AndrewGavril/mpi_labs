#!/bin/bash
for i in $(seq 1 15); do mpirun --hostfile hostfile -np $i python3 main.py >> res.csv; done