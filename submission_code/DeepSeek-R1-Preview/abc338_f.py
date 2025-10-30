def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    
    INF = 1 << 60
    dist = [[INF] * N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0
    
    for _ in range(M):
        u = int(input[idx]) - 1
        idx += 1
        v = int(input[idx]) - 1
        idx += 1
        w = int(input[idx])
        idx += 1
        if dist[u][v] > w:
            dist[u][v] = w
    
    # Floyd-Warshall algorithm
    for k in range(N):
        for i in range(N):
            if dist[i][k] == INF:
                continue
            for j in range(N):
                if dist[k][j] == INF:
                    continue
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    # Check if any node can reach all others
    found = False
    for u in range(N):
        all_reachable = True
        for v in range(N):
            if dist[u][v] == INF:
                all_reachable = False
                break
        if all_reachable:
            found = True
            break
    if not found:
        print("No")
        return
    
    # TSP DP
    full_mask = (1 << N) - 1
    dp = [[INF] * N for _ in range(1 << N)]
    
    for u in range(N):
        dp[1 << u][u] = 0
    
    for mask in range(1, 1 << N):
        for u in range(N):
            if not (mask & (1 << u)):
                continue
            current_cost = dp[mask][u]
            if current_cost == INF:
                continue
            for v in range(N):
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                new_cost = current_cost + dist[u][v]
                if new_cost < dp[new_mask][v]:
                    dp[new_mask][v] = new_cost
    
    min_total = min(dp[full_mask][u] for u in range(N))
    if min_total == INF:
        print("No")
    else:
        print(min_total)
    
if __name__ == '__main__':
    main()