import itertools

def solve():
    n, m, k = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))

    min_cost = float('inf')

    for i in range(1 << m):
        selected_edges = []
        for j in range(m):
            if (i >> j) & 1:
                selected_edges.append(edges[j])

        if len(selected_edges) != n - 1:
            continue

        adj = [[] for _ in range(n + 1)]
        for u, v, w in selected_edges:
            adj[u].append(v)
            adj[v].append(u)

        visited = [False] * (n + 1)
        q = [1]
        visited[1] = True
        count = 0
        while q:
            curr = q.pop(0)
            count += 1
            for neighbor in adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    q.append(neighbor)

        if count != n:
            continue

        cost = 0
        for _, _, w in selected_edges:
            cost = (cost + w) % k

        min_cost = min(min_cost, cost)

    print(min_cost)

solve()