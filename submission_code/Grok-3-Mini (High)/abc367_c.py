import sys
import itertools

# Read all input and parse integers
data = list(map(int, sys.stdin.read().split()))

# Extract N and K
N = data[0]
K = data[1]

# Extract the list of R values
R = data[2:2+N]

# Create a list of ranges for each position
ranges_list = [range(1, r + 1) for r in R]

# Generate all possible sequences using itertools.product
for seq in itertools.product(*ranges_list):
    # Check if the sum is a multiple of K
    if sum(seq) % K == 0:
        # Print the sequence with spaces
        print(*seq)