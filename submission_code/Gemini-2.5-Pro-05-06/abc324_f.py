import sys

def main():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())
    
    # Adjacency list: adj[u] stores list of (v, beauty, cost) for edges u -> v
    adj = [[] for _ in range(N + 1)] # 1-indexed, so N+1 size
    for _ in range(M):
        u, v, b, c = map(int, sys.stdin.readline().split())
        adj[u].append((v, b, c))

    # check(x_ratio) returns True if there exists a path P from 1 to N
    # such that Sum(b_e for e in P) / Sum(c_e for e in P) >= x_ratio.
    # This is equivalent to Sum(b_e - x_ratio * c_e for e in P) >= 0.
    # We find the path that maximizes this sum using dynamic programming.
    # If this maximum sum is >= 0, then such a path exists.
    def check(x_ratio: float) -> bool:
        # dp[i] stores the maximum sum of (beauty - x_ratio * cost) 
        # for any path from vertex 1 to vertex i.
        dp = [-float('inf')] * (N + 1)
        dp[1] = 0.0  # Base case: path from 1 to 1 has 0 "modified" weight.
        
        # Vertices are 1-indexed.
        # The condition u_i < v_i for all edges ensures that the graph is a DAG,
        # and processing vertices in increasing order of their indices (1, 2, ..., N)
        # is a valid topological sort.
        for u in range(1, N + 1):
            # If dp[u] is -infinity, vertex u is effectively not reachable from vertex 1
            # in a way that could contribute to the desired path.
            if dp[u] == -float('inf'):
                continue

            # For each outgoing edge from u
            for v, beauty, cost in adj[u]:
                # Calculate the modified weight for this edge
                modified_weight = float(beauty) - x_ratio * float(cost)
                
                # If path through u offers a better sum to reach v
                if dp[v] < dp[u] + modified_weight:
                    dp[v] = dp[u] + modified_weight
        
        # After iterating through all vertices and edges, dp[N] will hold the maximum
        # sum of modified weights for a path from 1 to N.
        # The problem guarantees that a path from 1 to N exists.
        # Thus, dp[N] will be a finite value (could be very negative if x_ratio is large).
        return dp[N] >= 0.0

    # Binary search for the maximum possible ratio.
    # Min ratio can be 0. Max can be 10000 (max_b/min_c = 10^4/1).
    low = 0.0
    high = 10000.0 
    # Using high = 10000.0. If answer is exactly 10000, check(10000) will be true, low becomes 10000.
    # Subsequent mid values will be tested. E.g. if initial high was 10001.0, mid = (10000+10001)/2 = 10000.5.
    # check(10000.5) will be false. high becomes 10000.5. The search converges to 10000.
    # If high is 10000.0, and answer is 10000.0, low becomes 10000.0. Mid becomes (10000.0+10000.0)/2=10000.0.
    # The loop continues but low doesn't change. This is fine.
    # For safety, often a slightly larger high is used, e.g., 10000.0 + epsilon, or 10001.0.
    # But 10000.0 is fine as max possible ratio for check.

    # Perform a fixed number of iterations for binary search.
    # 100 iterations are generally sufficient for typical precision requirements (e.g., 10^-9).
    for _ in range(100): 
        mid = (low + high) / 2.0
        # This break condition is for when interval size is limited by float precision.
        # For 100 iterations, it might not be strictly necessary but good practice.
        if mid == low or mid == high: # pragma: no cover
            break
            
        if check(mid):
            # If ratio 'mid' is achievable, it means 'mid' could be the answer,
            # or an even higher ratio might be possible. So, we set 'low = mid'.
            low = mid
        else:
            # If ratio 'mid' is not achievable, it's too high.
            # We need to search in the lower half: 'high = mid'.
            high = mid
            
    # 'low' will converge to the maximum ratio.
    # Print with required precision.
    sys.stdout.write(f"{low:.15f}
")

if __name__ == '__main__':
    main()