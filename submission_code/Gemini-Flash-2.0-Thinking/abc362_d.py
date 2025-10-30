import heapq

def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    edges = []
    for _ in range(m):
        u, v, b = map(int, input().split())
        edges.append((u - 1, v - 1, b))

    adj = [[] for _ in range(n)]
    for u, v, b in edges:
        adj[u].append((v, b))
        adj[v].append((u, b))

    results = []
    for target_node in range(1, n):
        dist = [float('inf')] * n
        dist[0] = a[0]
        pq = [(a[0], 0)]

        while pq:
            d, u = heapq.heappop(pq)

            if d > dist[u]:
                continue

            for v, weight in adj[u]:
                new_dist = dist[u] + weight + a[v]
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))

        results.append(dist[target_node])

    print(*results)

if __name__ == "__main__":
    solve()