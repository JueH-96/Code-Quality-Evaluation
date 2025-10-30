import itertools

n, k = map(int, input().split())
r = list(map(int, input().split()))

# Generate the ranges for each position
ranges = [range(1, ri + 1) for ri in r]

valid_sequences = []

# Iterate through all possible sequences
for seq in itertools.product(*ranges):
    if sum(seq) % k == 0:
        valid_sequences.append(seq)

# Print each valid sequence in lexicographical order
for s in valid_sequences:
    print(' '.join(map(str, s)))