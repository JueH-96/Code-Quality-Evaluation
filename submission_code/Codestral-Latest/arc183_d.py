import sys
from collections import deque, defaultdict

input = sys.stdin.read
data = input().split()

n = int(data[0])
edges = [tuple(map(int, data[i].split())) for i in range(1, n)]

# Build the tree
tree = defaultdict(list)
for a, b in edges:
    tree[a].append(b)
    tree[b].append(a)

# Find the depth of each node
def bfs(start):
    queue = deque([(start, 0)])
    depth = {}
    while queue:
        node, d = queue.popleft()
        depth[node] = d
        for neighbor in tree[node]:
            if neighbor not in depth:
                queue.append((neighbor, d + 1))
    return depth

depth = bfs(1)

# Find the diameter of the tree
def find_diameter():
    max_depth = max(depth, key=depth.get)
    depth_from_max = bfs(max_depth)
    diameter_end = max(depth_from_max, key=depth_from_max.get)
    return diameter_end, depth_from_max[diameter_end]

diameter_end, diameter_length = find_diameter()

# Find the path from the diameter end to the root
path = []
current = diameter_end
while current != 1:
    for neighbor in tree[current]:
        if neighbor in depth and depth[neighbor] == depth[current] - 1:
            path.append((current, neighbor))
            current = neighbor
            break

# Pair the leaves
pairs = []
for i in range(0, len(path), 2):
    pairs.append((path[i][0], path[i + 1][0]))

# Output the pairs
for x, y in pairs:
    print(x, y)