import sys
import heapq

def solve():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    
    roads = []
    for _ in range(M):
        A = int(data[index]) - 1
        index += 1
        B = int(data[index]) - 1
        index += 1
        C = int(data[index])
        index += 1
        roads.append((A, B, C))
    
    queries = []
    for _ in range(Q):
        t = int(data[index])
        index += 1
        if t == 1:
            i = int(data[index]) - 1
            index += 1
            queries.append((1, i))
        elif t == 2:
            x = int(data[index]) - 1
            index += 1
            y = int(data[index]) - 1
            index += 1
            queries.append((2, x, y))
    
    closed_roads = set()
    
    def dijkstra(start, end):
        dist = [float('inf')] * N
        dist[start] = 0
        pq = [(0, start)]
        
        while pq:
            current_dist, u = heapq.heappop(pq)
            
            if current_dist > dist[u]:
                continue
            
            for v, length in graph[u]:
                if (u, v) in closed_roads or (v, u) in closed_roads:
                    continue
                if dist[u] + length < dist[v]:
                    dist[v] = dist[u] + length
                    heapq.heappush(pq, (dist[v], v))
        
        return dist[end] if dist[end] != float('inf') else -1
    
    for query in queries:
        if query[0] == 1:
            closed_roads.add((roads[query[1]][0], roads[query[1]][1]))
        elif query[0] == 2:
            x, y = query[1], query[2]
            graph = [[] for _ in range(N)]
            for i, (a, b, c) in enumerate(roads):
                if (a, b) not in closed_roads and (b, a) not in closed_roads:
                    graph[a].append((b, c))
                    graph[b].append((a, c))
            result = dijkstra(x, y)
            print(result)