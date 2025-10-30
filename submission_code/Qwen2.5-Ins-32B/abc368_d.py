import sys
from collections import defaultdict

def find_lca(u, v, parent, depth):
    if depth[u] < depth[v]:
        u, v = v, u
    for i in range(20, -1, -1):
        if depth[u] - (1 << i) >= depth[v]:
            u = parent[u][i]
    if u == v:
        return u
    for i in range(20, -1, -1):
        if parent[u][i] != parent[v][i]:
            u = parent[u][i]
            v = parent[v][i]
    return parent[u][0]

def dfs(u, p, d):
    depth[u] = d
    parent[u][0] = p
    for i in range(1, 20):
        parent[u][i] = parent[parent[u][i-1]][i-1]
    for v in graph[u]:
        if v != p:
            dfs(v, u, d + 1)

def solve():
    N, K = map(int, input().split())
    graph = defaultdict(list)
    for _ in range(N-1):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
    vertices = list(map(int, input().split()))

    global depth, parent
    depth = [0] * (N + 1)
    parent = [[0] * 20 for _ in range(N + 1)]
    dfs(1, 1, 0)

    lca = vertices[0]
    for i in range(1, K):
        lca = find_lca(lca, vertices[i], parent, depth)

    visited = [False] * (N + 1)
    for v in vertices:
        u = v
        while u != lca:
            visited[u] = True
            u = parent[u][0]

    visited[lca] = True
    queue = [lca]
    count = 1
    while queue:
        u = queue.pop()
        for v in graph[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
                count += 1

    print(count)

solve()