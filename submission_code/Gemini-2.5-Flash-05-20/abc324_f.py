import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())

    # Adjacency list: adj[u] stores a list of (v, beauty, cost) tuples for edges from u.
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v, b, c = map(int, sys.stdin.readline().split())
        adj[u].append((v, b, c))

    # check(R) function:
    # Returns True if there exists a path from vertex 1 to vertex N such that
    # sum(b_i - R * c_i) >= 0.
    def check(R):
        # dp[i] stores the maximum sum of (b_j - R*c_j) for a path from vertex 1 to vertex i.
        # Initialize with negative infinity to denote unreachable vertices.
        dp = [-float('inf')] * (N + 1)
        dp[1] = 0.0  # Base case: path from 1 to 1 has a sum of 0.

        # Iterate through vertices from 1 to N. Since u_i < v_i for all edges,
        # this order processes nodes in topological order, ensuring that dp[u] is
        # finalized before it's used to update dp[v].
        for u in range(1, N + 1):
            # If vertex u is unreachable from 1, skip it as it cannot extend any valid path.
            if dp[u] == -float('inf'):
                continue
            
            # For each edge (u, v) originating from u
            for v, b, c in adj[u]:
                # Calculate the new weight for the current edge based on R
                current_edge_weight = float(b) - R * float(c)
                
                # Update dp[v] if a path through u gives a greater sum to v
                dp[v] = max(dp[v], dp[u] + current_edge_weight)
        
        # If dp[N] is non-negative, it means a path exists from 1 to N
        # where sum(b_i - R*c_i) >= 0, which implies sum(b_i)/sum(c_i) >= R.
        return dp[N] >= 0

    # Binary search for the maximum ratio R
    low = 0.0
    # A safe upper bound: max_beauty (10^4) / min_cost (1) = 10^4.
    # We use 10001.0 to be slightly above the maximum possible value.
    high = 10001.0 

    # Perform a fixed number of iterations for sufficient precision.
    # 100 iterations are typically enough for double precision and 10^-9 accuracy
    # over the given range.
    for _ in range(100):
        mid = (low + high) / 2
        if check(mid):
            low = mid  # mid is achievable or a lower value, so we can try a larger R
        else:
            high = mid # mid is too high, so R must be smaller

    # Print the result formatted to 16 decimal places for required precision.
    print(f"{low:.16f}")

solve()