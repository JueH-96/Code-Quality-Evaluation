# YOUR CODE HERE
n, m = map(int, input().split())
graph = {i: [] for i in range(1, n+1)}
for _ in range(m):
    a, b = map(int, input().split())
    graph[a].append(b)

def dfs(node, visited, path):
    visited[node] = True
    path.append(node)
    for neighbor in graph[node]:
        if not visited[neighbor]:
            if dfs(neighbor, visited, path):
                return True
        elif neighbor in path:
            return True
    path.pop()
    return False

def find_strongest():
    visited = [False] * (n + 1)
    path = []
    for i in range(1, n + 1):
        if not visited[i]:
            if dfs(i, visited, path):
                return -1
            if len(path) == n:
                return path[0]
            path.clear()
    return -1

print(find_strongest())