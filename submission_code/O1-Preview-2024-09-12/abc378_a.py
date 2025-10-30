# YOUR CODE HERE

import sys

A = list(map(int, sys.stdin.read().split()))

counts = [0] * 5  # Colors are from 1 to 4, we use index 1 to 4

for ai in A:
    counts[ai] += 1

total_pairs = sum(counts[i] // 2 for i in range(1, 5))

print(total_pairs)