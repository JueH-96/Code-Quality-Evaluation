import sys
import functools

# Read all input data
data = list(map(int, sys.stdin.read().split()))
index = 0
N = data[index]
index += 1

# Create distance matrix
dist = [[0 for _ in range(N)] for _ in range(N)]

# Read the weights and fill the distance matrix
for i in range(N - 1):  # i from 0 to N-2
    for j in range(i + 1, N):  # j from i+1 to N-1
        dist[i][j] = data[index]
        dist[j][i] = dist[i][j]  # Make it symmetric
        index += 1

# Define the DP function with memoization
@functools.lru_cache(None)
def max_match(mask):
    if mask == 0:
        return 0
    # Find the smallest i with bit set in mask
    for i in range(N):
        if mask & (1 << i):
            break
    # Option 1: leave i unmatched
    val_leave = max_match(mask ^ (1 << i))
    ans = val_leave
    # Option 2: match i to j > i with j in mask
    for j in range(i + 1, N):
        if mask & (1 << j):  # Check if j is in mask
            weight = dist[i][j]
            new_mask = mask ^ (1 << i) ^ (1 << j)
            val_match = weight + max_match(new_mask)
            if val_match > ans:
                ans = val_match
    return ans

# Compute the maximum weight matching for the full set
answer = max_match((1 << N) - 1)

# Output the answer
print(answer)