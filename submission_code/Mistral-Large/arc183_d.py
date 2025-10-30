import sys
from collections import deque, defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = []
index = 1
for _ in range(N-1):
    A = int(data[index])
    B = int(data[index + 1])
    edges.append((A, B))
    index += 2

# Build the tree using adjacency list
tree = defaultdict(list)
for A, B in edges:
    tree[A].append(B)
    tree[B].append(A)

# Function to find the distance between two vertices
def bfs(start, end):
    queue = deque([(start, 0)])
    visited = set()
    visited.add(start)
    while queue:
        current, dist = queue.popleft()
        if current == end:
            return dist
        for neighbor in tree[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1

# Find all leaves
leaves = [node for node in range(1, N+1) if len(tree[node]) == 1]

# Pair leaves to maximize the total score
pairings = []
while leaves:
    max_dist = -1
    best_pair = (0, 0)
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            dist = bfs(leaves[i], leaves[j])
            if dist > max_dist:
                max_dist = dist
                best_pair = (leaves[i], leaves[j])
    pairings.append(best_pair)
    leaves.remove(best_pair[0])
    leaves.remove(best_pair[1])

# Output the pairings
for pair in pairings:
    print(pair[0], pair[1])