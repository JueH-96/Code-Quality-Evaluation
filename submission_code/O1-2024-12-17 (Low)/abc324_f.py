def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    edges = [[] for _ in range(N+1)]
    idx = 2
    max_beauty = 0
    
    for _ in range(M):
        u = int(input_data[idx]); v = int(input_data[idx+1])
        b = int(input_data[idx+2]); c = int(input_data[idx+3])
        idx += 4
        edges[u].append((v, b, c))
        if b > max_beauty:
            max_beauty = b
    
    # We will apply binary search on the ratio X
    # We check feasibility by constructing the "weight" = b_i - X * c_i
    # and using DP over the DAG to find the maximum sum from 1 to N.
    # If dp[N] >= 0 for some X, then that X is feasible.
    
    def feasible(X):
        # DP array
        from math import inf
        dp = [-inf] * (N+1)
        dp[1] = 0.0
        
        # Relax edges in ascending order of vertices (DAG property: u < v)
        for u in range(1, N+1):
            if dp[u] == -inf:
                continue
            for (v, b, c) in edges[u]:
                val = dp[u] + (b - X*c)
                if val > dp[v]:
                    dp[v] = val
        
        return dp[N] >= 0
    
    # Binary search boundaries
    # The ratio cannot exceed (sum(b_i) / 1) in the absolute worst case,
    # but a simpler safe upper bound is 10001 (b_i up to 1e4).
    low, high = 0.0, 10001.0
    
    for _ in range(64):  # 64 iterations for ~1e-19 precision in worst case
        mid = (low + high) / 2
        if feasible(mid):
            low = mid
        else:
            high = mid
    
    # Print with appropriate precision
    print(f"{low:.9f}")

# Don't forget to call main()!
if __name__ == "__main__":
    main()