from collections import defaultdict

def dfs(graph, start, end, visited):
    if start == end:
        return True
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited and dfs(graph, neighbor, end, visited):
            return True
    return False

def is_graph_good(N, M, edges, K, pairs):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    for x, y in pairs:
        if dfs(graph, x, y, set()):
            return False
    return True

def solve():
    N, M = map(int, input().split())
    edges = []
    for _ in range(M):
        u, v = map(int, input().split())
        edges.append((u, v))
    K = int(input())
    pairs = []
    for _ in range(K):
        x, y = map(int, input().split())
        pairs.append((x, y))
    Q = int(input())
    for _ in range(Q):
        p, q = map(int, input().split())
        new_edges = edges + [(p, q)]
        if is_graph_good(N, M + 1, new_edges, K, pairs):
            print("Yes")
        else:
            print("No")

solve()