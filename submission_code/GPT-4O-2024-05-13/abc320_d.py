# YOUR CODE HERE
import sys
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

edges = defaultdict(list)
index = 2
for _ in range(M):
    A = int(data[index])
    B = int(data[index + 1])
    X = int(data[index + 2])
    Y = int(data[index + 3])
    edges[A].append((B, X, Y))
    edges[B].append((A, -X, -Y))
    index += 4

coordinates = [None] * (N + 1)
coordinates[1] = (0, 0)

queue = deque([1])
while queue:
    current = queue.popleft()
    current_x, current_y = coordinates[current]
    
    for neighbor, dx, dy in edges[current]:
        if coordinates[neighbor] is None:
            coordinates[neighbor] = (current_x + dx, current_y + dy)
            queue.append(neighbor)
        else:
            expected_x = current_x + dx
            expected_y = current_y + dy
            if coordinates[neighbor] != (expected_x, expected_y):
                coordinates[neighbor] = "undecidable"

for i in range(1, N + 1):
    if coordinates[i] is None or coordinates[i] == "undecidable":
        print("undecidable")
    else:
        print(coordinates[i][0], coordinates[i][1])