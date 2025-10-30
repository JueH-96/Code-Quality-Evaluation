import sys

def main():
    """
    Solves the problem of finding the minimum weight walk visiting all vertices.
    """
    # Use fast I/O
    input = sys.stdin.readline

    # Read N and M
    try:
        line = input()
        if not line.strip(): return
        N, M = map(int, line.split())
    except (IOError, ValueError):
        return

    # Read edges. Vertices are 1-based, convert to 0-based.
    edges = []
    for _ in range(M):
        u, v, w = map(int, input().split())
        edges.append((u - 1, v - 1, w))

    # Use a large number to represent infinity
    inf = float('inf')

    # Step 1: All-pairs shortest paths using Floyd-Warshall
    # dist[i][j] will store the shortest path cost from vertex i to j.
    dist = [[inf] * N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0
    
    for u, v, w in edges:
        dist[u][v] = w
        
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist[i][k] != inf and dist[k][j] != inf:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    # Step 2: TSP with Bitmask DP
    # dp[mask][u]: min cost to visit vertices in `mask`, ending at `u`.
    dp = [[inf] * N for _ in range(1 << N)]

    # Base cases: A path visiting a single vertex has cost 0. We can start at any vertex.
    for i in range(N):
        dp[1 << i][i] = 0

    # Iterate through all subsets of vertices (masks)
    for mask in range(1, 1 << N):
        # For each vertex `u` in the current subset `mask`
        for u in range(N):
            if (mask >> u) & 1:
                # If a path to this state (mask, u) has been found
                if dp[mask][u] != inf:
                    # Try to extend the path to a new vertex `v` not yet in `mask`
                    for v in range(N):
                        if not ((mask >> v) & 1):
                            # Check if there is a path from u to v
                            if dist[u][v] != inf:
                                new_mask = mask | (1 << v)
                                new_cost = dp[mask][u] + dist[u][v]
                                # Update the cost for the new state (new_mask, v)
                                dp[new_mask][v] = min(dp[new_mask][v], new_cost)

    # Step 3: Find the result
    # The final mask represents visiting all N vertices.
    final_mask = (1 << N) - 1
    # The result is the minimum of dp[final_mask][u] over all possible ending vertices `u`.
    min_cost = min(dp[final_mask])

    # Output the result
    if min_cost == inf:
        print("No")
    else:
        # The costs are integers, so the result should be an integer.
        print(int(min_cost))

if __name__ == "__main__":
    main()