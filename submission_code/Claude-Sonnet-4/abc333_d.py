from collections import deque, defaultdict

n = int(input())

# Build adjacency list
graph = defaultdict(list)
for _ in range(n - 1):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)

# BFS from vertex 1 to find maximum distance to any vertex
visited = [False] * (n + 1)
queue = deque([(1, 0)])  # (vertex, distance)
visited[1] = True
max_distance = 0

while queue:
    vertex, distance = queue.popleft()
    max_distance = max(max_distance, distance)
    
    for neighbor in graph[vertex]:
        if not visited[neighbor]:
            visited[neighbor] = True
            queue.append((neighbor, distance + 1))

print(max_distance)