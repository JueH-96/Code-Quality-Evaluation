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
        distances = {i: float('inf') for i in range(n)}
        distances[0] = a[0]
        pq = [(a[0], 0)]

        while pq:
            dist, u = heapq.heappop(pq)

            if dist > distances[u]:
                continue

            for v, weight in adj[u]:
                new_dist = dist + weight + a[v]
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
        results.append(distances[target_node])
    print(*results)

solve()