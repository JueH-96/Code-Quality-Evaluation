import heapq

def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    edges = []
    for _ in range(m):
        u, v, b = map(int, input().split())
        edges.append((u - 1, v - 1, b))
        edges.append((v - 1, u - 1, b))

    dist = [float('inf')] * n
    dist[0] = 0
    pq = [(0, 0)]

    while pq:
        d, u = heapq.heappop(pq)

        if d > dist[u]:
            continue

        for v1, v2, b in edges:
            if v1 == u:
                cost = d + a[u] + b + a[v2] - a[v2]
                if cost < dist[v2]:
                    dist[v2] = cost
                    heapq.heappush(pq, (cost, v2))
    
    ans = []
    for i in range(1,n):
        ans.append(dist[i] + a[i])

    print(*ans)

solve()