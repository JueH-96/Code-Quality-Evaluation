import heapq

def solve():
    n, m, q = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))
    
    closed_edges = set()

    def dijkstra(start, end):
        dist = {i: float('inf') for i in range(1, n + 1)}
        dist[start] = 0
        pq = [(0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            
            for i, (v1, v2, w) in enumerate(edges):
                if i + 1 in closed_edges:
                    continue
                
                if v1 == u:
                    v = v2
                elif v2 == u:
                    v = v1
                else:
                    continue
                
                if dist[v] > dist[u] + w:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))
        
        return dist[end] if dist[end] != float('inf') else -1

    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            closed_edges.add(query[1])
        else:
            x, y = query[1], query[2]
            print(dijkstra(x, y))

solve()