from collections import defaultdict

def dfs(graph, node, parent, visited):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor != parent:
            if neighbor in visited:
                return [node, neighbor]
            else:
                path = dfs(graph, neighbor, node, visited)
                if path:
                    return path
    return None

def count_valid_graphs(N, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    count = 0
    for i in range(1, N):
        for j in range(i+1, N+1):
            if j not in graph[i]:
                visited = set()
                cycle = dfs(graph, i, None, visited)
                if cycle:
                    if all(len(graph[node]) == 3 for node in cycle):
                        count += 1
    return count

# Read input
N = int(input())
edges = []
for _ in range(N-1):
    u, v = map(int, input().split())
    edges.append((u, v))

# Solve the problem
result = count_valid_graphs(N, edges)
print(result)