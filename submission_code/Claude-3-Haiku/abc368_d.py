from collections import defaultdict

def dfs(graph, node, visited, required):
    visited.add(node)
    if node in required:
        return 1
    count = 0
    for neighbor in graph[node]:
        if neighbor not in visited:
            count += dfs(graph, neighbor, visited, required)
    return count

def solve():
    n, k = map(int, input().split())
    graph = defaultdict(list)
    for _ in range(n-1):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
    required = set(map(int, input().split()))
    visited = set()
    return dfs(graph, 1, visited, required)

print(solve())