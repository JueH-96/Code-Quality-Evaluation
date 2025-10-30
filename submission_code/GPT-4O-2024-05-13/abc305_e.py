# YOUR CODE HERE
import sys
import collections

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])

edges = []
index = 3
for _ in range(M):
    a = int(data[index])
    b = int(data[index + 1])
    edges.append((a, b))
    index += 2

guards = []
for _ in range(K):
    p = int(data[index])
    h = int(data[index + 1])
    guards.append((p, h))
    index += 2

# Create adjacency list for the graph
graph = collections.defaultdict(list)
for a, b in edges:
    graph[a].append(b)
    graph[b].append(a)

# BFS to find all guarded vertices
guarded = set()
for p, h in guards:
    queue = collections.deque([(p, 0)])
    visited = set()
    while queue:
        current, dist = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        if dist <= h:
            guarded.add(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))

# Output the result
guarded = sorted(guarded)
print(len(guarded))
print(" ".join(map(str, guarded)))