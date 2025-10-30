# YOUR CODE HERE
def solve():
    n1, n2, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))

    adj = [[] for _ in range(n1 + n2 + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def bfs(start, end, graph):
        q = [(start, 0)]
        visited = {start}
        while q:
            curr, dist = q.pop(0)
            if curr == end:
                return dist
            for neighbor in graph[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, dist + 1))
        return -1

    max_d = 0
    for u in range(1, n1 + 1):
        for v in range(n1 + 1, n1 + n2 + 1):
            temp_adj = [list(neighbors) for neighbors in adj]
            temp_adj[u].append(v)
            temp_adj[v].append(u)
            
            d = bfs(1, n1 + n2, temp_adj)
            max_d = max(max_d, d)

    print(max_d)

solve()