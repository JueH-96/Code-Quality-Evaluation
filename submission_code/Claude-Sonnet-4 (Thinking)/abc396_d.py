from collections import defaultdict

n, m = map(int, input().split())

graph = defaultdict(list)

for _ in range(m):
    u, v, w = map(int, input().split())
    graph[u].append((v, w))
    graph[v].append((u, w))

min_xor = float('inf')

def dfs(node, visited, current_xor):
    global min_xor
    
    if node == n:
        min_xor = min(min_xor, current_xor)
        return
    
    for neighbor, weight in graph[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            dfs(neighbor, visited, current_xor ^ weight)
            visited.remove(neighbor)

visited = {1}
dfs(1, visited, 0)

print(min_xor)