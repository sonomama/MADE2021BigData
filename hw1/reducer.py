#!/usr/bin/env python3 

import sys

count = 0
mean = 0
for line in sys.stdin:
    line = line.strip()
    mean_i, count_i = line.split(" ")
    mean_i = float(mean_i)
    count_i = int(count_i)
    mean = (count_i * mean_i + count * mean) / (count_i + count)
    count += count_i

print(mean)
