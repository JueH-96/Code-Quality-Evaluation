import sys

# Read input
N, M, C, K = map(int, input().split())
A = list(map(int, input().split()))

# Compute the answer
total = 0
for k in range(K):
    min_val = float('inf')
    for i in range(N):
        min_val = min(min_val, (C * k + A[i]) % M)
    total += min_val

# Print the answer
print(total)