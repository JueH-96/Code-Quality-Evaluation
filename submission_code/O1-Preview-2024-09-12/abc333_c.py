# YOUR CODE HERE
import sys
from itertools import combinations_with_replacement

lv = [int('1' * i) for i in range(1, 16)]  # Generate repunits up to length 15
combos = combinations_with_replacement(lv, 3)
sums = set(sum(c) for c in combos)
sums_sorted = sorted(sums)

N = int(sys.stdin.readline())
print(sums_sorted[N - 1])