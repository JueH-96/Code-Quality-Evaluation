from collections import defaultdict, deque

N, M = map(int, input().split())

# Build the graph
graph = defaultdict(list)

for _ in range(M):
    A, B, X, Y = map(int, input().split())
    graph[A].append((B, X, Y))
    graph[B].append((A, -X, -Y))

# BFS to determine coordinates
coordinates = {}
coordinates[1] = (0, 0)  # Person 1 is at the origin

queue = deque([1])

while queue:
    current = queue.popleft()
    current_x, current_y = coordinates[current]
    
    for neighbor, dx, dy in graph[current]:
        if neighbor not in coordinates:
            new_x, new_y = current_x + dx, current_y + dy
            coordinates[neighbor] = (new_x, new_y)
            queue.append(neighbor)

# Output results
for i in range(1, N + 1):
    if i in coordinates:
        print(coordinates[i][0], coordinates[i][1])
    else:
        print("undecidable")