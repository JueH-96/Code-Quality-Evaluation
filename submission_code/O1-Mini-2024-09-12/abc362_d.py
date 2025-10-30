import sys
import threading
import heapq

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        U, V, B = map(int, sys.stdin.readline().split())
        adj[U].append((V, B))
        adj[V].append((U, B))
    INF = 1 << 60
    dist = [INF] * (N+1)
    dist[1] = A[0]
    heap = [(dist[1], 1)]
    visited = [False]*(N+1)
    while heap:
        current_dist, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        for v, B in adj[u]:
            if not visited[v]:
                new_dist = current_dist + B + A[v-1]
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(heap, (dist[v], v))
    print(' '.join(str(dist[i]) for i in range(2, N+1)))

threading.Thread(target=main).start()