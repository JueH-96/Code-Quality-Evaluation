import sys

# Function to check if a given ratio lambda_val is achievable (or better)
def check(lambda_val, N, adj):
    """
    Computes the maximum path weight from vertex 1 to vertex N
    using edge weights b_i - lambda_val * c_i.
    Returns True if the maximum path weight is non-negative, False otherwise.
    """
    # dp[i] will store the maximum path weight from vertex 1 to vertex i
    # Initialize with negative infinity, except for the start node (vertex 1)
    dp = [-float('inf')] * (N + 1)
    dp[1] = 0

    # Iterate through vertices in topological order (1 to N).
    # Since edge constraint is u_i < v_i, this is a valid topological order.
    for u in range(1, N + 1):
        # If dp[u] is still -inf, it means vertex u is unreachable from vertex 1
        # (considering the current weights and paths found so far).
        # We can skip processing edges from such a vertex as no path from 1 can extend through it.
        if dp[u] == -float('inf'):
            continue

        # Iterate through edges starting from u
        for v, b, c in adj[u]:
            # Calculate the weight of edge (u, v) for the current lambda value
            weight = b - lambda_val * c
            # Relax the edge: if a path through u to v is better than the current best path to v, update dp[v].
            dp[v] = max(dp[v], dp[u] + weight)

    # The problem guarantees a path from 1 to N exists.
    # After processing all reachable vertices and their outgoing edges,
    # dp[N] will hold the maximum weight of a path from 1 to N.
    # We check if this maximum weight is non-negative.
    # Use a small tolerance for floating point comparison near zero if needed,
    # but for binary search, a direct comparison is often fine.
    # The binary search itself handles the convergence around the root.
    return dp[N] >= 0

# Main execution block
if __name__ == "__main__":
    # Read input
    N, M = map(int, sys.stdin.readline().split())

    # Build adjacency list representation of the graph
    # adj[u] will store a list of tuples (v, b, c) for edges from u to v
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v, b, c = map(int, sys.stdin.readline().split())
        adj[u].append((v, b, c))

    # Binary search for the maximum ratio (lambda)
    # We are looking for the maximum lambda such that check(lambda) is True.
    # The possible values of lambda (the ratio) are non-negative.
    # A loose upper bound for lambda is max(b_i) / min(c_i), which is 10^4 / 1 = 10^4.
    # The maximum ratio cannot exceed the maximum edge ratio in the optimal path.
    # So, max ratio <= max(b_i/c_i) <= 10^4.
    # We use a slightly larger upper bound for safety, e.g., 10001.
    low = 0.0
    high = 10001.0

    # Perform binary search for a fixed number of iterations to achieve required precision.
    # 100 iterations are more than sufficient for a range of ~10^4 and 10^-9 precision.
    # The width of the search interval is halved in each iteration.
    # After K iterations, the interval width is (high - low) / 2^K.
    # To get precision epsilon, we need (high - low) / 2^K <= epsilon.
    # For initial range ~10^4 and epsilon ~10^-9, K needs to be around log2(10^13) approx 44.
    # 100 iterations is very safe.
    for _ in range(100):
        mid = (low + high) / 2.0
        if check(mid, N, adj):
            # If check(mid) is True, it means there exists a path P from 1 to N
            # with TotalBeauty(P) / TotalCost(P) >= mid.
            # So, the maximum ratio is at least mid. We try searching in the upper half.
            low = mid
        else:
            # If check(mid) is False, it means for all paths P from 1 to N,
            # TotalBeauty(P) / TotalCost(P) < mid.
            # So, mid is too high. We need to search in the lower half.
            high = mid

    # After the binary search loop, 'low' (or 'high') contains the approximate maximum ratio.
    # Print the result with the required precision (at least 9 decimal places).
    print(f"{low:.12f}") # Print with 12 decimal places for safety/standard practice