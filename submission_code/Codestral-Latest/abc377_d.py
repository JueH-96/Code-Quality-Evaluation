# YOUR CODE HERE
import sys
from itertools import accumulate

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

intervals = []
for i in range(N):
    L_i = int(data[2 * i + 2])
    R_i = int(data[2 * i + 3])
    intervals.append((L_i, R_i))

# Sort intervals by their ending points
intervals.sort(key=lambda x: x[1])

# Calculate the number of valid pairs for each interval
valid_pairs = [0] * (M + 2)
for L, R in intervals:
    valid_pairs[L] += 1
    valid_pairs[R + 1] -= 1

# Calculate the cumulative sum
cumulative_sum = list(accumulate(valid_pairs))

# Calculate the number of valid pairs (l, r)
total_valid_pairs = 0
for l in range(1, M + 1):
    for r in range(l, M + 1):
        if cumulative_sum[l] == cumulative_sum[r + 1]:
            total_valid_pairs += 1

print(total_valid_pairs)