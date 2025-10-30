from collections import deque

def solve():
    n1, n2, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))

    def bfs(start, end, added_edge=None):
        n = n1 + n2
        adj = [[] for _ in range(n + 1)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        if added_edge:
            u, v = added_edge
            adj[u].append(v)
            adj[v].append(u)

        q = deque([(start, 0)])
        visited = {start}
        while q:
            curr, dist = q.popleft()
            if curr == end:
                return dist
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, dist + 1))
        return float('inf')

    max_d = 0
    for u in range(1, n1 + 1):
        for v in range(n1 + 1, n1 + n2 + 1):
            d = bfs(1, n1 + n2, (u, v))
            max_d = max(max_d, d)

    print(max_d)

solve()