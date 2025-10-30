def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    
    # Build adjacency list: edges[u] = list of (v, beauty, cost)
    edges = [[] for _ in range(N+1)]
    idx = 2
    for _ in range(M):
        u = int(input_data[idx]); v = int(input_data[idx+1])
        b = int(input_data[idx+2]); c = int(input_data[idx+3])
        idx += 4
        edges[u].append((v, b, c))
    
    # Check feasibility of a candidate ratio "mid" via DAG DP
    def feasible(mid):
        from math import inf
        dist = [-inf]*(N+1)
        dist[1] = 0.0
        for u in range(1, N+1):
            if dist[u] == -inf:
                continue
            base = dist[u]
            for (v, b, c) in edges[u]:
                alt = base + (b - mid*c)
                if alt > dist[v]:
                    dist[v] = alt
        return dist[N] >= 0
    
    # Binary search for the maximum ratio
    left, right = 0.0, 10000.0
    for _ in range(50):  # 50 iterations -> ~1e-15 precision
        mid = (left + right) / 2
        if feasible(mid):
            left = mid
        else:
            right = mid
    
    # Print the result with sufficient precision
    print(f"{(left + right)/2:.12f}")

# Do not forget to call main()
main()