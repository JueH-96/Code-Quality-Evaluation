from collections import deque

def solve():
    n1, n2, m = map(int, input().split())
    adj = [[] for _ in range(n1 + n2 + 1)]
    for _ in range(m):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)

    def bfs(start_node, allowed_nodes):
        distances = {}
        if start_node not in allowed_nodes:
            return distances

        queue = deque([(start_node, 0)])
        distances[start_node] = 0

        while queue:
            u, dist = queue.popleft()
            for v in adj[u]:
                if v in allowed_nodes and v not in distances:
                    distances[v] = dist + 1
                    queue.append((v, dist + 1))
        return distances

    dist1 = bfs(1, set(range(1, n1 + 1)))
    dist2 = bfs(n1 + n2, set(range(n1 + 1, n1 + n2 + 1)))

    max_d = 0
    for u in range(1, n1 + 1):
        for v in range(n1 + 1, n1 + n2 + 1):
            if u in dist1 and v in dist2:
                max_d = max(max_d, dist1[u] + 1 + dist2[v])

    print(max_d)

if __name__ == "__main__":
    solve()