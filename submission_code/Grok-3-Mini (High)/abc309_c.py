import sys
import bisect

# Read all input from stdin
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
K = int(data[index])
index += 1

# Read medicines and store as list of tuples (a, b)
meds = []
for _ in range(N):
    a = int(data[index])
    index += 1
    b = int(data[index])
    index += 1
    meds.append((a, b))

# Sort the medicines by a in ascending order
meds.sort()

# Create a_list (sorted list of a values)
a_list = [med[0] for med in meds]

# Create b_list (list of b values in the same order as a_list)
b_list = [med[1] for med in meds]

# Compute suffix sum of b_list
len_m = N
suffix_sum = [0] * len_m
if N > 0:
    suffix_sum[len_m - 1] = b_list[len_m - 1]
    for i in range(len_m - 2, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + b_list[i]

# Find max_a
max_a = a_list[-1] if N > 0 else 0  # Though N >= 1, handle just in case

# Binary search to find the smallest D where sum of pills <= K
left = 1
right = max_a + 1
result = -1
while left <= right:
    mid = (left + right) // 2
    # Find the index where a >= mid starts
    idx = bisect.bisect_left(a_list, mid)
    if idx < N:
        sum_D = suffix_sum[idx]
    else:
        sum_D = 0
    if sum_D <= K:
        result = mid
        right = mid - 1  # Try to find a smaller D
    else:
        left = mid + 1  # Need a larger D

# Output the result
print(result)