import sys
from itertools import accumulate

input = sys.stdin.readline

# Parse input
N, K = map(int, input().split())
S = input().strip()

# Convert string to list of integers
S_list = [int(c) for c in S]

# Find transitions from 0 to 1 and 1 to 0
transitions = [i for i in range(1, N) if S[i - 1] != S[i]]

# Combine transitions to find the start and end of 1-blocks
block_boundaries = []
for i, t in enumerate(transitions):
    if i % 2 == 0:
        block_boundaries.append(t)
    if i % 2 == 1:
        block_boundaries.append(t - 1)

# Match given 'K' with the correct 1-block indexes
idx = transitions[(K - 1) * 2]

# Calculate the length of the 1-block to move
length = (idx - block_boundaries[K - 1]) + 1

# Perform the moving operation as described
S_list = S_list[:idx+1] + S_list[block_boundaries[K-1]:block_boundaries[K-1] + length] + S_list[idx+1:block_boundaries[K-1]] + S_list[block_boundaries[K-1] + length:]
S_modified = ''.join(map(str, S_list))

print(S_modified)