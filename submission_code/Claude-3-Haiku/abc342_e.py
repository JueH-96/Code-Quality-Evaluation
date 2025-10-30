from collections import defaultdict
from heapq import heappush, heappop

def solve():
    N, M = map(int, input().split())
    trains = []
    for _ in range(M):
        l, d, k, c, A, B = map(int, input().split())
        for t in range(l, l + k * d, d):
            trains.append((t, t + c, A, B))
    
    # Dijkstra's algorithm
    graph = defaultdict(list)
    for t, c, A, B in trains:
        graph[A].append((t, c, B))
    
    ans = []
    for s in range(1, N):
        pq = [(0, s)]
        dist = [float('inf')] * (N + 1)
        dist[s] = 0
        while pq:
            d, u = heappop(pq)
            if d > dist[u]:
                continue
            if u == N:
                ans.append(d)
                break
            for t, c, v in graph[u]:
                new_d = d + c
                if new_d < dist[v]:
                    dist[v] = new_d
                    heappush(pq, (new_d, v))
        else:
            ans.append(-float('inf'))
    
    for f in ans:
        if f == -float('inf'):
            print('Unreachable')
        else:
            print(f)

solve()