# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = []
for i in range(N - 1):
    A = int(data[2 * i + 1])
    B = int(data[2 * i + 2])
    edges.append((A, B))

C = list(map(int, data[N * 2 + 1:]))

# Build the tree
from collections import defaultdict, deque
tree = defaultdict(list)
for A, B in edges:
    tree[A].append(B)
    tree[B].append(A)

# Function to perform DFS and calculate the sum of distances
def dfs(node, parent):
    total = 0
    for neighbor in tree[node]:
        if neighbor != parent:
            total += dfs(neighbor, node)
    distances[node] = total
    return distances[node] + C[node - 1]

# Initialize distances array
distances = [0] * (N + 1)

# Perform DFS starting from node 1
dfs(1, -1)

# Calculate the minimum value of f(x)
min_value = float('inf')
for i in range(1, N + 1):
    min_value = min(min_value, distances[i])

# Print the result
print(min_value)