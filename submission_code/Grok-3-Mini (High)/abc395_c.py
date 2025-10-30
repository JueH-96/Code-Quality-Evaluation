import sys
from collections import defaultdict

# Read input from stdin
data = sys.stdin.read().split()
N = int(data[0])
A = list(map(int, data[1:]))

# Create a dictionary to store indices for each value in A
idx_dict = defaultdict(list)
for i in range(N):
    idx_dict[A[i]].append(i)

# Initialize minimum length to infinity
min_len = float('inf')

# Iterate through each list of indices in the dictionary
for idx_list in idx_dict.values():
    if len(idx_list) >= 2:
        # Find the minimum difference between consecutive indices
        min_diff = min(idx_list[i + 1] - idx_list[i] for i in range(len(idx_list) - 1))
        # The length of the subarray is min_diff + 1
        min_len_for_this = min_diff + 1
        # Update the overall minimum length
        if min_len_for_this < min_len:
            min_len = min_len_for_this

# Check if any subarray with a duplicate was found
if min_len == float('inf'):
    print(-1)
else:
    print(min_len)