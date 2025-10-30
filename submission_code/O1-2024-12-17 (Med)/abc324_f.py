def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    edges_input = input_data[2:]
    
    # Build adjacency list
    # adjacency[u] = [(v, b, c), ...] for edges u -> v with beauty b and cost c
    adjacency = [[] for _ in range(N+1)]
    idx = 0
    for _ in range(M):
        u = int(edges_input[idx]); v = int(edges_input[idx+1])
        b = float(edges_input[idx+2]); c = float(edges_input[idx+3])
        idx += 4
        adjacency[u].append((v, b, c))

    # Function to check feasibility for a given ratio x
    # We compute the maximum possible sum of (b_i - x*c_i)
    # over all paths from 1 to N in this DAG. If dp[N] >= 0, it's feasible.
    def feasible(x):
        from math import inf
        dp = [-inf]*(N+1)
        dp[1] = 0.0
        # The graph is acyclic in ascending vertex order (u < v)
        for u in range(1, N+1):
            if dp[u] == -inf:
                continue
            base = dp[u]
            for (v, b, c) in adjacency[u]:
                val = base + (b - x*c)
                if val > dp[v]:
                    dp[v] = val
        return dp[N] >= 0

    # Binary search for the maximum ratio
    left, right = 0.0, 10001.0
    for _ in range(60):
        mid = (left + right) / 2
        if feasible(mid):
            left = mid
        else:
            right = mid

    # Print the answer with sufficient precision
    print(f"{left:.15f}")

# Do not forget to call main()
if __name__ == "__main__":
    main()