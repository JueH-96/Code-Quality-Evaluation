def main():
    import sys
    import math
    import heapq
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    # Build an adjacency list for stages 1..N
    # For each stage i (1<= i <= N-1), we have two edges:
    #   1) from i to i+1 with cost A[i]
    #   2) from i to X[i] with cost B[i]
    graph = [[] for _ in range(N+1)]
    idx = 1
    for i in range(1, N):
        A = int(data[idx])
        B = int(data[idx+1])
        X = int(data[idx+2])
        idx += 3
        # Option 1: clear stage i and unlock stage i+1
        graph[i].append((i+1, A))
        # Option 2: clear stage i and unlock stage X
        graph[i].append((X, B))
    
    # Use Dijkstra's algorithm from stage 1 to stage N.
    dist = [math.inf] * (N+1)
    dist[1] = 0
    heap = [(0, 1)]  # (time_so_far, stage)
    
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        if u == N:
            # Found the minimum cost to reach stage N.
            print(d)
            return
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
                
    print(dist[N])
    
if __name__ == "__main__":
    main()