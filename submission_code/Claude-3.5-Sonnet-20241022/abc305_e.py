from collections import deque

def bfs(graph, start, n):
    dist = [-1] * (n+1)
    dist[start] = 0
    queue = deque([start])
    
    while queue:
        v = queue.popleft()
        for u in graph[v]:
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                queue.append(u)
                
    return dist

# Read input
N, M, K = map(int, input().split())

# Create adjacency list
graph = [[] for _ in range(N+1)]
for _ in range(M):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

# Read guard positions and stamina
guards = []
for _ in range(K):
    p, h = map(int, input().split())
    guards.append((p, h))

# For each guard, run BFS to get distances to all vertices
guarded = set()
for p, h in guards:
    distances = bfs(graph, p, N)
    # Add vertices within stamina range to guarded set
    for v in range(1, N+1):
        if distances[v] != -1 and distances[v] <= h:
            guarded.add(v)

# Sort guarded vertices
guarded = sorted(list(guarded))

# Print output
print(len(guarded))
print(*guarded)