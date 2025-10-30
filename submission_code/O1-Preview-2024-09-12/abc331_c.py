# YOUR CODE HERE
import sys
N,*rest=map(int, sys.stdin.read().split())
A = rest[:N]
from collections import defaultdict
freq = defaultdict(int)
total_sum = 0
for a in A:
    freq[a] += 1
    total_sum += a
unique_Ai = sorted(freq.keys())
cumulative_sum = 0
sum_over_greater = {}
for Ai in unique_Ai:
    sum_over_greater[Ai] = total_sum - cumulative_sum - Ai * freq[Ai]
    cumulative_sum += Ai * freq[Ai]
result = [sum_over_greater[a] for a in A]
print(' '.join(map(str, result)))