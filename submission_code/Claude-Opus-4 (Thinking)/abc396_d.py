def dfs(graph, current, target, visited, current_xor, min_xor):
    if current == target:
        return min(min_xor, current_xor)
    
    visited[current] = True
    for neighbor, weight in graph[current]:
        if not visited[neighbor]:
            min_xor = dfs(graph, neighbor, target, visited, current_xor ^ weight, min_xor)
    visited[current] = False
    
    return min_xor

# Read input
N, M = map(int, input().split())
graph = [[] for _ in range(N + 1)]

for _ in range(M):
    u, v, w = map(int, input().split())
    graph[u].append((v, w))
    graph[v].append((u, w))

# Find minimum XOR path from 1 to N
visited = [False] * (N + 1)
min_xor = dfs(graph, 1, N, visited, 0, float('inf'))

print(min_xor)