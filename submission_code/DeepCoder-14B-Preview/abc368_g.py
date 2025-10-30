import itertools

# Read input
n, k = map(int, input().split())
r = list(map(int, input().split()))

# Generate all possible sequences using itertools.product
ranges = [range(1, ri + 1) for ri in r]
sequences = itertools.product(*ranges)

# Collect valid sequences
valid = []
for seq in sequences:
    if sum(seq) % k == 0:
        valid.append(seq)

# Print each valid sequence in the required format
for seq in valid:
    print(' '.join(map(str, seq)))