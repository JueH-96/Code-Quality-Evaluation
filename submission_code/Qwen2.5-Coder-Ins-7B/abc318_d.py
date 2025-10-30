import sys
from itertools import combinations

# Read input
input = sys.stdin.read
data = input().split()

# Number of vertices
N = int(data[0])

# Distance matrix
D = [[0] * N for _ in range(N)]
index = 1
for i in range(N):
    for j in range(i + 1, N):
        D[i][j] = int(data[index])
        D[j][i] = D[i][j]
        index += 1

# Function to calculate the total weight of a set of edges
def total_weight(edges):
    return sum(D[i][j] for i, j in edges)

# Generate all possible sets of edges
max_weight = 0
for r in range(1, N // 2 + 1):
    for edges in combinations(range(N), 2 * r):
        edges = [(edges[i], edges[i + 1]) for i in range(0, 2 * r, 2)]
        max_weight = max(max_weight, total_weight(edges))

# Output the result
print(max_weight)