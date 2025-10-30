import sys
from itertools import combinations

input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

# Create a list of all sock colors
socks = [i for i in range(1, N+1) if i not in A]

# Calculate the number of pairs we need to form
num_pairs = (2*N - K) // 2

# If the number of remaining socks is odd, we will have one sock left over
leftover = (2*N - K) % 2

# Initialize the minimum weirdness to a large value
min_weirdness = float('inf')

# Generate all possible pairs of socks
for pair in combinations(socks, 2*num_pairs):
    # Calculate the weirdness for this combination of pairs
    weirdness = sum(abs(pair[i] - pair[i+1]) for i in range(0, 2*num_pairs, 2))
    # Update the minimum weirdness if this combination is better
    min_weirdness = min(min_weirdness, weirdness)

print(min_weirdness)