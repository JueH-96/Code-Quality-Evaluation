import sys
import itertools

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = [0] * N
B = [0] * N
for i in range(N):
    A[i] = int(data[index])
    index += 1
    B[i] = int(data[index])
    index += 1

# Find all edges where cards can be paired
edges = []
for i in range(N):
    for j in range(i + 1, N):
        if A[i] == A[j] or B[i] == B[j]:
            edges.append((i, j))

# DP to compute Grundy numbers
dp_size = 1 << N
dp = [None] * dp_size

# Iterate over all subset sizes and compute Grundy numbers
for size in range(0, N + 1):
    for combo in itertools.combinations(range(N), size):
        # Build mask for the current subset
        mask = 0
        for card in combo:
            mask |= (1 << card)
        
        # Find all possible moves and their Grundy numbers
        outcomes = set()
        for edge in edges:
            i, j = edge
            if (mask & (1 << i)) and (mask & (1 << j)):
                new_mask = mask & ~((1 << i) | (1 << j))
                outcomes.add(dp[new_mask])
        
        # Compute mex
        mex_val = 0
        while mex_val in outcomes:
            mex_val += 1
        
        # Set the Grundy number for this mask
        dp[mask] = mex_val

# The full mask has all cards
full_mask = (1 << N) - 1
grundy_full = dp[full_mask]

# Determine the winner
if grundy_full != 0:
    print("Takahashi")
else:
    print("Aoki")