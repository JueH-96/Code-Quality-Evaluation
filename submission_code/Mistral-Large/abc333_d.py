import sys
from collections import deque, defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = []

for i in range(1, N):
    u, v = int(data[2*i]), int(data[2*i+1])
    edges.append((u, v))

# Build the tree using adjacency list
tree = defaultdict(list)
for u, v in edges:
    tree[u].append(v)
    tree[v].append(u)

# BFS to find the minimum number of operations to delete vertex 1
queue = deque([1])
visited = set([1])
leaf_count = 0

while queue:
    current = queue.popleft()
    if len(tree[current]) <= 1 or (len(tree[current]) == 2 and 1 in tree[current]):
        leaf_count += 1
    for neighbor in tree[current]:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)

print(leaf_count)