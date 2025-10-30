import sys

# Read input
N = int(input())
S = input()
C = list(map(int, input().split()))

# Convert the cost array to an array of tuples (cost, digit)
costs = [(c if s == '0' else 0, s == '0') for s, c in zip(S, C)]

# Add the 'flip' cost and digit for the transition between each pair of digits, and add a dummy at the end
costs = [(0, False)] + [(c1 + c2, d1 ^ d2) for (c1, d1), (c2, d2) in zip(costs, costs[1:])] + [(0, False)]

# Find the minimum cost to either the same prefix or have a flip at the current index
dp = [0, 0]
for c, d in costs:
    dp_new = [c + min(dp), min(dp) + (c if d else 0)]
    dp = dp_new

# The answer is the minimum cost to reach either state (same prefix or flip) without having a last flip
print(min(dp))