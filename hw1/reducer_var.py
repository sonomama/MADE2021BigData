#!/usr/bin/env python3 

import sys

count = mean = var = 0
for line in sys.stdin:
    line = line.strip()
    count_i, mean_i, var_i = line.split(" ")
    count_i = int(count_i)
    mean_i, var_i = float(mean_i), float(var_i)

    tot_count = count_i + count
    var = (count_i * var_i + count * var) / tot_count + \
          count_i * count * ((mean_i - mean) / tot_count) ** 2
    mean = (count_i * mean_i + count * mean) / tot_count
    
    count += count_i

print(var)
