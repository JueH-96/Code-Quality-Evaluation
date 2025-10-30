def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # Read input
    it = iter(input_data)
    n = int(next(it))
    m = int(next(it))
    graph = [[] for _ in range(n + 1)]
    # Build the graph: each edge goes from u to v with beauty b and cost c.
    # The property u < v guarantees that the graph is a Directed Acyclic Graph.
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        b = int(next(it))
        c = int(next(it))
        graph[u].append((v, b, c))
    
    # We'll use a binary search to find the maximum ratio R such that there is a path
    # from vertex 1 to vertex n with total beauty / total cost >= R.
    # Define for a candidate ratio "mid", value on an edge becomes b - mid * c.
    # For a valid path, the sum over edges in that path is:
    #   total_value = (sum of b) - mid * (sum of c)
    # If there exists a path with total_value >= 0 then the ratio is achievable.
    #
    # Because the graph is a DAG (given u < v for every edge), we can use dynamic programming in order.
    
    lo = 0.0           # lower bound of possible ratio
    hi = 1e4           # upper bound; since b_i <= 1e4 and c_i >= 1, the ratio cannot exceed 1e4.
    eps_iterations = 80  # 80 iterations guarantee high precision (error <= 1e-9)
    
    for _ in range(eps_iterations):
        mid = (lo + hi) / 2.0
        # dp[v] will store the maximum possible sum (b - mid*c) over all paths from vertex 1 to v.
        dp = [-10**18] * (n + 1)
        dp[1] = 0.0
        
        # Process vertices in increasing order (this is a valid topological order)
        for u in range(1, n + 1):
            if dp[u] == -10**18:
                continue  # skip unreachable vertex
            for v, b, c in graph[u]:
                candidate = dp[u] + b - mid * c
                if candidate > dp[v]:
                    dp[v] = candidate
        # If dp[n] >= 0, it is possible to achieve this ratio, so we try a higher candidate.
        if dp[n] >= 0:
            lo = mid
        else:
            hi = mid

    # Print the maximum ratio with enough precision; the answer is accepted if error <= 1e-9.
    sys.stdout.write("{:.16f}".format(lo))


if __name__ == '__main__':
    main()