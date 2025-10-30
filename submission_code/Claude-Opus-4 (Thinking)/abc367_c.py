from itertools import product

# Read input
n, k = map(int, input().split())
r_list = list(map(int, input().split()))

# Generate all possible sequences
ranges = [range(1, r + 1) for r in r_list]

# Check each sequence and print if sum is divisible by k
for seq in product(*ranges):
    if sum(seq) % k == 0:
        print(' '.join(map(str, seq)))