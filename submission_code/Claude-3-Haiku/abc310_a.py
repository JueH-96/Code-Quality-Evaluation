import sys

# Read input
N, P, Q = map(int, input().split())
D = list(map(int, input().split()))

# Calculate minimum total payment
min_total = float('inf')
for i in range(N):
    total = Q + D[i]
    min_total = min(min_total, total)
min_total = min(min_total, P)

print(min_total)