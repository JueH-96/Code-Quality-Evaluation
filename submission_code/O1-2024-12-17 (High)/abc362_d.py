def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    idx = 0
    
    N = int(input_data[idx]); idx += 1
    M = int(input_data[idx]); idx += 1
    
    A = list(map(int, input_data[idx:idx+N]))
    idx += N
    
    # Build adjacency list
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u = int(input_data[idx]); idx += 1
        v = int(input_data[idx]); idx += 1
        b = int(input_data[idx]); idx += 1
        # Switch to 0-based indexing
        u -= 1
        v -= 1
        # Undirected graph
        adj[u].append((v, b))
        adj[v].append((u, b))
    
    INF = float('inf')
    dist = [INF]*N
    dist[0] = A[0]  # cost to include the first vertex
    
    # Dijkstra
    pq = [(A[0], 0)]  # (distance, node)
    while pq:
        current_dist, u = heapq.heappop(pq)
        if current_dist > dist[u]:
            continue
        for (v, edge_w) in adj[u]:
            # Cost to v = dist[u] + edge_w + A[v]
            cost = dist[u] + edge_w + A[v]
            if cost < dist[v]:
                dist[v] = cost
                heapq.heappush(pq, (cost, v))
    
    # Print results for vertices 2..N (i.e. dist[1..N-1])
    print(' '.join(str(dist[i]) for i in range(1, N)))

# Do not forget to call main() at the end
main()