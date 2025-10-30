import sys
import heapq

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N+1)]
    edges = []
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
    
    INF = float('inf')
    dist = [[INF] * (K + 1) for _ in range(N + 1)]
    dist[1][0] = 0
    heap = []
    heapq.heappush(heap, (0, 1, 0))
    
    while heap:
        current_dist, u, cnt = heapq.heappop(heap)
        if u == N:
            print(current_dist)
            return
        if current_dist > dist[u][cnt]:
            continue
        for v in adj[u]:
            # Case 1: do not flip this edge
            if dist[v][cnt] > current_dist:
                dist[v][cnt] = current_dist
                heapq.heappush(heap, (dist[v][cnt], v, cnt))
            # Case 2: flip this edge, if cnt < K
            if cnt < K:
                if dist[v][cnt + 1] > current_dist + 1:
                    dist[v][cnt + 1] = current_dist + 1
                    heapq.heappush(heap, (dist[v][cnt + 1], v, cnt + 1))
    
    result = min(dist[N][k] for k in range(K + 1))
    print(result)

solve()