import sys
from itertools import product

# Read input
N, K = map(int, input().split())
R = list(map(int, input().split()))

# Generate all possible sequences
sequences = product(*(range(1, r + 1) for r in R))

# Filter sequences whose sum is a multiple of K
valid_sequences = [seq for seq in sequences if sum(seq) % K == 0]

# Sort sequences lexicographically
valid_sequences.sort()

# Print the valid sequences
for seq in valid_sequences:
    print(' '.join(map(str, seq)))