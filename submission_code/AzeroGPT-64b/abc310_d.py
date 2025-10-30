from functools import lru_cache

def generate_graph(n, m, edges):
    graph = [[] for _ in range(n)]
    for a, b in edges:
        graph[a-1].append(b-1)
        graph[b-1].append(a-1)
    return graph

def valid_coloring(graph, n, t, coloring, node):
    for neighbor in graph[node]:
        if coloring[neighbor] == coloring[node]:
            return False
    return True

def backtrack(graph, n, t, coloring, k):
    if k == n:
        return 1
    count = 0
    for color in range(t):
        if valid_coloring(graph, n, t, coloring, k):
            coloring[k] = color
            count += backtrack(graph, n, t, coloring, k + 1)
            coloring[k] = -1
    return count

def solve(N, T, M, edges):
    graph = generate_graph(N, M, edges)
    coloring = [-1] * N
    return backtrack(graph, N, T, coloring, 0)

# Read input from stdin
N, T, M = map(int, input().split())
edges = [tuple(map(int, input().split())) for _ in range(M)]

# Solve and write the answer to stdout
print(solve(N, T, M, edges))