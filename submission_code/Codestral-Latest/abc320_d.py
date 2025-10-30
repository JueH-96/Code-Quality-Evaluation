import sys
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2

edges = defaultdict(list)
for _ in range(M):
    A = int(data[index]) - 1
    B = int(data[index + 1]) - 1
    X = int(data[index + 2])
    Y = int(data[index + 3])
    edges[A].append((B, X, Y))
    index += 4

def bfs(start):
    queue = deque([(start, 0, 0)])
    visited = set()
    while queue:
        node, x, y = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor, dx, dy in edges[node]:
            if (neighbor, x + dx, y + dy) not in visited:
                queue.append((neighbor, x + dx, y + dy))
    return visited

def find_coordinates():
    coordinates = [None] * N
    for i in range(N):
        if coordinates[i] is None:
            visited = bfs(i)
            if len(visited) == 1:
                coordinates[i] = (0, 0)
            else:
                for node, x, y in visited:
                    if coordinates[node] is None:
                        coordinates[node] = (x, y)
                    elif coordinates[node] != (x, y):
                        coordinates[node] = "undecidable"
    return coordinates

coordinates = find_coordinates()
for coord in coordinates:
    if coord == "undecidable":
        print("undecidable")
    else:
        print(coord[0], coord[1])