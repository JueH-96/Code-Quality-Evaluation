N, K = map(int, input().split())
A = list(map(int, input().split()))

# Try all possible combinations of K elements to remove
min_diff = float('inf')

# For each subset of K elements to remove
from itertools import combinations
for to_remove in combinations(range(N), K):
    # Create remaining sequence after removing K elements
    remaining = []
    for i in range(N):
        if i not in to_remove:
            remaining.append(A[i])
            
    # Calculate max - min of remaining sequence
    diff = max(remaining) - min(remaining)
    min_diff = min(min_diff, diff)

print(min_diff)