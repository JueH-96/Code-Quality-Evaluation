def solve():
    import sys

    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    edges = []
    idx = 2
    adj = [[] for _ in range(N+1)]
    
    # Read edges
    for _ in range(M):
        u = int(input_data[idx]); v = int(input_data[idx+1])
        b = float(input_data[idx+2]); c = float(input_data[idx+3])
        idx += 4
        # Store adjacency for the DAG
        adj[u].append((v, b, c))
    
    # Since u < v for all edges, the graph is a DAG in ascending order of vertices.
    # We'll do a binary search on the ratio R = sum(b_i)/sum(c_i).
    # The maximum possible ratio is at most 10^4 (since b_i <= 10^4, c_i >= 1).
    
    def can_achieve_ratio(ratio):
        # DP to find the maximum sum of (b_i - ratio*c_i) along a path from 1 to N
        # If dp[N] >= 0, then we can achieve the ratio.
        dp = [-float('inf')] * (N+1)
        dp[1] = 0.0
        for v in range(1, N+1):
            if dp[v] == -float('inf'):
                continue
            base = dp[v]
            for (nx, b, c) in adj[v]:
                val = base + (b - ratio*c)
                if val > dp[nx]:
                    dp[nx] = val
        return dp[N] >= 0
    
    low, high = 0.0, 10000.0
    for _ in range(60):  # enough iterations for 1e-9 precision
        mid = (low + high) / 2
        if can_achieve_ratio(mid):
            low = mid
        else:
            high = mid
    
    print(f"{low:.9f}")

def main():
    solve()

if __name__ == "__main__":
    main()