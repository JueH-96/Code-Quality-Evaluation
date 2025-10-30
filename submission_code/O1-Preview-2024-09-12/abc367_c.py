# YOUR CODE HERE
import sys
import itertools

# Read input
N, K = map(int, sys.stdin.readline().split())
R = list(map(int, sys.stdin.readline().split()))

# Generate possible ranges for each position
ranges = [range(1, R_i + 1) for R_i in R]

# Generate all possible sequences
all_sequences = itertools.product(*ranges)

# Filter sequences where sum is multiple of K
valid_sequences = [seq for seq in all_sequences if sum(seq) % K == 0]

# Sort sequences lexicographically
valid_sequences.sort()

# Output the sequences
for seq in valid_sequences:
    print(' '.join(map(str, seq)))