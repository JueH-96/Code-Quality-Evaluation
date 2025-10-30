import sys
import sys
import sys
def solve():
    import sys, sys
    import sys
    from collections import defaultdict
    import sys
    import sys
    import sys
    from heapq import heappush, heappop

    def dijkstra(start, adj, N):
        INF = 1 << 60
        dist = [INF] * (N + 1)
        count = [0] * (N + 1)
        dist[start] = 0
        count[start] = 1
        heap = []
        heappush(heap, (0, start))
        while heap:
            d, u = heappop(heap)
            if d > dist[u]:
                continue
            for v, c in adj[u]:
                if dist[v] > d + c:
                    dist[v] = d + c
                    count[v] = count[u]
                    heappush(heap, (dist[v], v))
                elif dist[v] == d + c:
                    count[v] += count[u]
        return dist, count

    import sys

    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr]); ptr +=1
    M = int(data[ptr]); ptr +=1
    roads = []
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        A = int(data[ptr]); ptr +=1
        B = int(data[ptr]); ptr +=1
        C = int(data[ptr]); ptr +=1
        roads.append( (A, B, C) )
        adj[A].append( (B, C) )
        adj[B].append( (A, C) )
    
    dist1, counts1 = dijkstra(1, adj, N)
    distn, countsn = dijkstra(N, adj, N)
    D = dist1[N]
    total_counts = counts1[N]
    result = []
    for road in roads:
        u, v, c = road
        usage = 0
        if dist1[u] + c + distn[v] == D:
            usage += counts1[u] * countsn[v]
        if dist1[v] + c + distn[u] == D:
            usage += counts1[v] * countsn[u]
        if usage == total_counts:
            result.append("Yes")
        else:
            result.append("No")
    print('
'.join(result))