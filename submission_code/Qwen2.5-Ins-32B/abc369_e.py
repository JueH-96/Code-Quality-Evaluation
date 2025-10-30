import heapq
import sys

def dijkstra(graph, start, end):
    dist = [float('inf')] * len(graph)
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == end:
            return d
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return float('inf')

def solve():
    input = sys.stdin.read
    data = input().split()
    
    n, m = int(data[0]), int(data[1])
    idx = 2
    bridges = []
    for _ in range(m):
        u, v, t = int(data[idx]) - 1, int(data[idx + 1]) - 1, int(data[idx + 2])
        bridges.append((u, v, t))
        idx += 3
    
    q = int(data[idx])
    idx += 1
    queries = []
    for _ in range(q):
        k = int(data[idx])
        idx += 1
        bridges_needed = [int(data[idx + i]) - 1 for i in range(k)]
        queries.append(bridges_needed)
        idx += k
    
    for bridges_needed in queries:
        graph = [[] for _ in range(n)]
        for i, (u, v, t) in enumerate(bridges):
            if i in bridges_needed:
                graph[u].append((v, t))
                graph[v].append((u, t))
            else:
                graph[u].append((v, 0))
                graph[v].append((u, 0))
        
        min_time = float('inf')
        for i in bridges_needed:
            u, v, t = bridges[i]
            graph[u].append((v, t))
            graph[v].append((u, t))
            time = dijkstra(graph, 0, n - 1)
            min_time = min(min_time, time)
            graph[u].pop()
            graph[v].pop()
        
        print(min_time)

solve()