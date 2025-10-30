import heapq

def main():
    import sys
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N + 1)]  # 1-based indexing

    for i in range(1, N):
        A, B, X = map(int, sys.stdin.readline().split())
        adj[i].append((i + 1, A))
        adj[i].append((X, B))
    
    INF = float('inf')
    dist = [INF] * (N + 1)
    dist[1] = 0
    heap = []
    heapq.heappush(heap, (0, 1))
    
    while heap:
        d, u = heapq.heappop(heap)
        if u == N:
            print(d)
            return
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if dist[v] > d + w:
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))
    
    # If unreachable, but according to problem statement, it's always possible?
    print(-1)

if __name__ == "__main__":
    main()