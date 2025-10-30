import heapq

def solve():
    n, m, k = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))

    def shortest_path(chosen_edges):
        dist = {i: float('inf') for i in range(1, n + 1)}
        dist[1] = 0
        pq = [(0, 1)]

        while pq:
            d, u = heapq.heappop(pq)

            if d > dist[u]:
                continue

            for i, (start, end) in enumerate(edges):
                if start == u:
                    weight = 1 if i in chosen_edges else 0
                    if dist[end] > dist[u] + weight:
                        dist[end] = dist[u] + weight
                        heapq.heappush(pq, (dist[end], end))

        return dist[n]

    max_dist = 0
    for i in range(1 << m):
        chosen_edges = []
        count = 0
        for j in range(m):
            if (i >> j) & 1:
                chosen_edges.append(j)
                count += 1

        if count == k:
            dist = shortest_path(chosen_edges)
            max_dist = max(max_dist, dist)

    print(int(max_dist))

solve()