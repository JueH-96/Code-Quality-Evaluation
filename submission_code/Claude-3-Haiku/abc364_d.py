import sys

# Read input
N, Q = map(int, input().split())
A = list(map(int, input().split()))
queries = []
for _ in range(Q):
    b, k = map(int, input().split())
    queries.append((b, k))

# Solve each query
for b, k in queries:
    distances = [abs(a - b) for a in A]
    distances.sort()
    print(distances[k - 1])