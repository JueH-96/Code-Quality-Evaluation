def solve():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    # Edges adjacency list: for each stage i, store edges (next_stage, time_cost)
    edges = [[] for _ in range(N+1)]
    
    idx = 1
    for i in range(1, N):
        A_i = int(input_data[idx]); B_i = int(input_data[idx+1]); X_i = int(input_data[idx+2])
        idx += 3
        # From stage i, we can go to i+1 with cost A_i
        edges[i].append((i+1, A_i))
        # From stage i, we can go to X_i with cost B_i
        edges[i].append((X_i, B_i))
    
    # Dijkstra's algorithm to find the minimum cost to reach stage N
    INF = 10**18
    dist = [INF]*(N+1)
    dist[1] = 0
    pq = [(0,1)]  # (cost, stage)
    
    while pq:
        cost, u = heapq.heappop(pq)
        if cost > dist[u]:
            continue
        if u == N:
            # We have reached stage N with minimal cost
            print(cost)
            return
        for v, w in edges[u]:
            nc = cost + w
            if nc < dist[v]:
                dist[v] = nc
                heapq.heappush(pq, (nc, v))
    
    # If for some reason stage N is unreachable (problem statement suggests it is reachable),
    # dist[N] would remain INF. But let's just print dist[N] anyway.
    print(dist[N])

# Call solve() after defining it
solve()