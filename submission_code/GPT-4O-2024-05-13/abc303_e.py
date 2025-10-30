# YOUR CODE HERE
import sys
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N-1)]

# Create adjacency list
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# Find all leaf nodes
leaf_nodes = [node for node in adj if len(adj[node]) == 1]

# BFS to find the levels of stars
visited = set()
level_counts = []

queue = deque(leaf_nodes)
while queue:
    node = queue.popleft()
    if node in visited:
        continue
    visited.add(node)
    parent = None
    for neighbor in adj[node]:
        if neighbor not in visited:
            parent = neighbor
            break
    if parent is not None:
        leaf_count = 0
        for neighbor in adj[parent]:
            if neighbor in visited:
                leaf_count += 1
        level_counts.append(leaf_count)
        queue.append(parent)

level_counts.sort()
print(" ".join(map(str, level_counts)))