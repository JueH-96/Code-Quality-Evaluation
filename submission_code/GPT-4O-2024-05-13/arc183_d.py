# YOUR CODE HERE
import sys
input = sys.stdin.read
from collections import defaultdict, deque

data = input().split()
N = int(data[0])
edges = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N-1)]

# Build the adjacency list
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# Function to find the distance between two nodes in a tree using BFS
def bfs_distance(start, end):
    queue = deque([(start, 0)])
    visited = set()
    while queue:
        node, dist = queue.popleft()
        if node == end:
            return dist
        visited.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                queue.append((neighbor, dist + 1))
    return -1

# Find all leaves
leaves = [node for node in range(1, N+1) if len(adj[node]) == 1]

# Sort leaves to maximize distance
leaves.sort()

# Perform the operations
operations = []
while leaves:
    leaf1 = leaves.pop()
    leaf2 = leaves.pop(0)
    operations.append((leaf1, leaf2))
    # Remove the leaves from the tree
    for neighbor in adj[leaf1]:
        adj[neighbor].remove(leaf1)
        if len(adj[neighbor]) == 1:
            leaves.append(neighbor)
    for neighbor in adj[leaf2]:
        adj[neighbor].remove(leaf2)
        if len(adj[neighbor]) == 1:
            leaves.append(neighbor)
    leaves.sort()

# Output the operations
for x, y in operations:
    print(x, y)