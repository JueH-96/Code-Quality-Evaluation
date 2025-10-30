# YOUR CODE HERE
import sys

def solve():
    """
    This function reads input, solves the problem, and prints the answer.
    """
    try:
        input_stream = sys.stdin.readline
        N, M, K = map(int, input_stream().split())
        edges = [tuple(map(int, input_stream().split())) for _ in range(M)]
    except (IOError, ValueError):
        # Handle potential empty input on some platforms
        return

    # Create a reverse adjacency list. rev_adj[v] lists all u such that u -> v.
    rev_adj = [[] for _ in range(N + 1)]
    for u, v in edges:
        rev_adj[v].append(u)

    def check(d):
        """
        Check if it's possible to make the shortest path from 1 to N have length at least d,
        using at most K edges with weight 1.
        """
        if d == 0:
            return True

        # dp[i][v]: Minimum cost (number of edges to set to weight 1) to ensure
        # that all paths from vertex 1 to vertex v have a length of at least i.
        
        # If a cost exceeds K, it's too expensive. K+1 is a sufficient sentinel.
        inf = K + 1
        
        dp = [[inf] * (N + 1) for _ in range(d + 1)]

        # Base case: A path of length >= 0 requires 0 cost for any vertex.
        for v in range(N + 1):
            dp[0][v] = 0

        # Main DP calculation. Iterate through required path lengths i.
        for i in range(1, d + 1):
            # For each length i, we compute dp[i] using values from dp[i-1].
            # The recurrence for dp[i][v] depends on other dp[i][u] values,
            # forming a system of equations. We solve this using value iteration.
            
            # N iterations are sufficient for values to propagate across any simple path.
            for _ in range(N): 
                updated_in_pass = False
                for v in range(1, N + 1):
                    # No paths from 1 can reach 1 with length > 0 within the problem's scope.
                    if v == 1:
                        continue
                    
                    current_cost = 0
                    # The cost for vertex v is the sum of costs for each incoming edge.
                    for u in rev_adj[v]:
                        # For an edge u -> v, we have two choices:
                        # 1. Don't select it (cost 0): Paths to u must have length >= i. Cost: dp[i][u].
                        # 2. Select it (cost 1): Paths to u must have length >= i-1. Cost: 1 + dp[i-1][u].
                        # We take the minimum cost option for this edge.
                        edge_cost = min(dp[i][u], 1 + dp[i-1][u])
                        current_cost += edge_cost
                        
                        # Optimization: if cost exceeds K, we can cap it.
                        if current_cost > K:
                            current_cost = inf
                            break
                    
                    if current_cost < dp[i][v]:
                        dp[i][v] = current_cost
                        updated_in_pass = True

                if not updated_in_pass:
                    # If no values changed, we have converged.
                    break
        
        # The result of the check is whether the cost for vertex N is within our budget K.
        return dp[d][N] <= K

    # Binary search for the maximum possible shortest distance.
    low = 0
    # The maximum possible distance is at most K (if we use all K edges on one path)
    # and at most N-1 for simple paths. K is a safe upper bound.
    high = K + 1 
    ans = 0

    while low < high:
        mid = (low + high) // 2
        if check(mid):
            # If we can achieve distance 'mid', it's a potential answer.
            # Try for a larger distance.
            ans = mid
            low = mid + 1
        else:
            # If we can't achieve 'mid', we can't achieve anything larger either.
            # Try for a smaller distance.
            high = mid
            
    print(ans)

solve()