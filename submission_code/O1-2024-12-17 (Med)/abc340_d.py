def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A, B, X = [], [], []
    idx = 1
    for _ in range(N - 1):
        A.append(int(input_data[idx]))
        B.append(int(input_data[idx + 1]))
        X.append(int(input_data[idx + 2]))
        idx += 3

    # Build adjacency list
    adj = [[] for _ in range(N + 1)]
    for i in range(1, N):
        # i -> i+1 with cost A[i-1]
        adj[i].append((i + 1, A[i - 1]))
        # i -> X[i-1] with cost B[i-1]
        adj[i].append((X[i - 1], B[i - 1]))

    # Dijkstra's algorithm
    INF = 10**20
    dist = [INF] * (N + 1)
    dist[1] = 0
    pq = [(0, 1)]  # (cost, node)

    while pq:
        current_dist, node = heapq.heappop(pq)
        if current_dist > dist[node]:
            continue
        if node == N:
            print(current_dist)
            return
        for nxt, cost in adj[node]:
            new_dist = current_dist + cost
            if new_dist < dist[nxt]:
                dist[nxt] = new_dist
                heapq.heappush(pq, (new_dist, nxt))

    print(dist[N])

# Don't forget to call main()
if __name__ == "__main__":
    main()