from collections import defaultdict

def dfs(graph, node, visited, color):
    visited[node] = color
    for neighbor in graph[node]:
        if neighbor not in visited:
            if not dfs(graph, neighbor, visited, 1 - color):
                return False
        elif visited[neighbor] == color:
            return False
    return True

def solve():
    N, M = map(int, input().split())
    graph = defaultdict(list)
    for _ in range(M):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)

    visited = {}
    for node in range(1, N+1):
        if node not in visited:
            if not dfs(graph, node, visited, 0):
                print("Takahashi")
                return

    print("Aoki")

solve()