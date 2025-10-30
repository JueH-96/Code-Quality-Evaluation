def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    A = [0] + list(map(int, input_data[2:2+N]))
    edges_data = input_data[2+N:]
    
    adj = [[] for _ in range(N+1)]
    idx = 0
    for _ in range(M):
        u = int(edges_data[idx]); v = int(edges_data[idx+1]); b = int(edges_data[idx+2])
        idx += 3
        adj[u].append((v, b))
        adj[v].append((u, b))
    
    INF = 10**20
    dist = [INF]*(N+1)
    dist[1] = A[1]  # Cost to reach vertex 1 includes A[1]
    
    # Dijkstra's algorithm
    pq = [(dist[1], 1)]
    while pq:
        cur_cost, u = heapq.heappop(pq)
        if cur_cost > dist[u]:
            continue
        for v, cost_edge in adj[u]:
            # Moving from u to v has additional cost = B_(u,v) + A[v]
            new_cost = cur_cost + cost_edge + A[v]
            if new_cost < dist[v]:
                dist[v] = new_cost
                heapq.heappush(pq, (new_cost, v))
    
    # Print results for vertices 2..N
    print(' '.join(str(dist[i]) for i in range(2, N+1)))


# Do not forget to call main
if __name__ == "__main__":
    main()