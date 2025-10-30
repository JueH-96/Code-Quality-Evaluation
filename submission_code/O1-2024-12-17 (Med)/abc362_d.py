def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    A = list(map(int, input_data[2:2+N]))

    edges_data = input_data[2+N:]
    idx = 0

    # Build adjacency list
    # We'll store for each u: (v, cost = B + A[v]), and vice versa.
    adjacency = [[] for _ in range(N)]
    for _ in range(M):
        u = int(edges_data[idx]); v = int(edges_data[idx+1]); b = int(edges_data[idx+2])
        idx += 3
        u -= 1; v -= 1  # switch to 0-based
        adjacency[u].append((v, b + A[v]))
        adjacency[v].append((u, b + A[u]))

    # Dijkstra from vertex 0 (which is vertex 1 in 1-based)
    INF = float('inf')
    dist = [INF]*N
    dist[0] = A[0]  # include the weight of the start vertex
    pq = [(dist[0], 0)]
    visited = [False]*N

    while pq:
        cur_dist, u = heapq.heappop(pq)
        if dist[u] < cur_dist:
            continue
        visited[u] = True
        for (v, cost_uv) in adjacency[u]:
            new_dist = cur_dist + cost_uv
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))

    # Print distances from vertex 1 to vertices 2..N
    # dist[i] is the cost to get to vertex i (0-based) = i+1(1-based)
    print(' '.join(str(dist[i]) for i in range(1, N)))

# Do not forget to call main()
main()