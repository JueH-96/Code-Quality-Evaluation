import heapq

def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    edges = []
    for _ in range(m):
        u, v, b = map(int, input().split())
        edges.append((u - 1, v - 1, b))

    ans = []
    for i in range(1, n):
        pq = [(a[0], 0)]
        visited = {0}
        min_dist = float('inf')

        while pq:
            dist, u = heapq.heappop(pq)

            if u == i:
                min_dist = dist
                break

            for edge in edges:
                v1, v2, weight = edge
                v = -1
                if v1 == u:
                    v = v2
                elif v2 == u:
                    v = v1
                
                if v != -1 and v not in visited:
                    heapq.heappush(pq, (dist + weight + a[v], v))
                    visited.add(v)

        ans.append(min_dist)

    print(*ans)

solve()