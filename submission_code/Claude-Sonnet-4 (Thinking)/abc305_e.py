from collections import deque

# Read inputs
N, M, K = map(int, input().split())

# Build the graph
graph = [[] for _ in range(N + 1)]
for _ in range(M):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

guards = []
for _ in range(K):
    p, h = map(int, input().split())
    guards.append((p, h))

# Find all guarded vertices
guarded = set()

for p, h in guards:
    # BFS from position p up to distance h
    visited = [False] * (N + 1)
    queue = deque([(p, 0)])  # (vertex, distance)
    visited[p] = True
    
    while queue:
        vertex, dist = queue.popleft()
        
        # If this vertex is within the guard's range, mark it as guarded
        if dist <= h:
            guarded.add(vertex)
        
        # If we haven't reached the maximum distance, explore neighbors
        if dist < h:
            for neighbor in graph[vertex]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, dist + 1))

# Sort the guarded vertices
guarded_list = sorted(list(guarded))

# Output
print(len(guarded_list))
if len(guarded_list) > 0:
    print(' '.join(map(str, guarded_list)))