import sys
from collections import deque, defaultdict

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1

edges = []
for _ in range(N - 1):
    A, B = int(data[index]), int(data[index + 1])
    edges.append((A, B))
    index += 2

C = [int(data[index + i]) for i in range(N)]

# Build the tree
tree = defaultdict(list)
for A, B in edges:
    tree[A].append(B)
    tree[B].append(A)

# Calculate the sum of C values
total_sum = sum(C)

# BFS to calculate the sum of distances from each node to all other nodes
def bfs(start):
    queue = deque([start])
    visited = [False] * (N + 1)
    visited[start] = True
    distance_sum = 0
    level = 0
    while queue:
        level_size = len(queue)
        for _ in range(level_size):
            node = queue.popleft()
            distance_sum += level * C[node - 1]
            for neighbor in tree[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        level += 1
    return distance_sum

# Calculate f(v) for each vertex v
min_f = float('inf')
for v in range(1, N + 1):
    f_v = bfs(v)
    min_f = min(min_f, f_v)

print(min_f)