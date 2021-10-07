#!/usr/bin/env python3 

import sys

count = val = 0
for row, line in enumerate(sys.stdin):
    line = line.strip("\t")
    columns = line.split(",")
    if len(columns) < 7 or not columns[-7].isdigit():
        continue
    count += 1
    val += int(columns[-7])

print(val/count, count)