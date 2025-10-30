# YOUR CODE HERE
import heapq

def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    edges = []
    for _ in range(m):
        u, v, b = map(int, input().split())
        edges.append((u - 1, v - 1, b))

    results = []
    for i in range(1, n):
        min_weight = float('inf')
        
        q = [(a[0], 0)]
        visited = [False] * n
        dist = [float('inf')] * n
        dist[0] = a[0]

        while q:
            d, u = heapq.heappop(q)

            if visited[u]:
                continue
            visited[u] = True
            
            if u == i:
                min_weight = min(min_weight, d)

            for u1, v1, b1 in edges:
                if u1 == u and not visited[v1]:
                    new_dist = d + b1 + a[v1]
                    if new_dist < dist[v1]:
                        dist[v1] = new_dist
                        heapq.heappush(q, (new_dist, v1))
                elif v1 == u and not visited[u1]:
                    new_dist = d + b1 + a[u1]
                    if new_dist < dist[u1]:
                        dist[u1] = new_dist
                        heapq.heappush(q, (new_dist, u1))
        results.append(min_weight)

    print(*results)

solve()