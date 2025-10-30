import sys

def solve():
    # Read N (number of vertices) and M (number of edges)
    N, M = map(int, sys.stdin.readline().split())

    # Adjacency list to represent the graph.
    # adj[u] will store a list of (v, weight) tuples for edges from u to v.
    # Vertices are 1-indexed in input, so we convert to 0-indexed (0 to N-1) for array indexing.
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        adj[u-1].append((v-1, w)) # Adjust to 0-indexed

    # dp[mask][last_vertex] stores the minimum total weight of a walk
    # that visits exactly the vertices represented by 'mask' and ends at 'last_vertex'.
    # Initialize with a very large value (infinity) to represent unreachable states.
    INF = float('inf')
    dp = [[INF] * N for _ in range(1 << N)] # (1 << N) is 2^N

    # Base cases:
    # A walk starting at vertex i (0-indexed) has visited only vertex i
    # and has a total weight of 0. The mask for this state is (1 << i).
    for i in range(N):
        dp[1 << i][i] = 0

    # Iterate through all possible masks (subsets of vertices visited).
    # The masks are processed in increasing order of their integer value.
    # This ensures that when we compute dp[new_mask][v], the dp[mask][u] values
    # (for mask <= new_mask) are either already finalized or in the process of being
    # finalized in a way that respects the shortest path property.
    for mask in range(1, 1 << N):
        for u in range(N):
            # If a path to (mask, u) doesn't exist or is currently infinity, skip.
            # This check implicitly handles cases where u is not part of the mask's visited set
            # for that path, as such states would remain INF unless explicitly set by a valid path.
            if dp[mask][u] == INF:
                continue

            # Try to extend the walk from vertex u to its neighbors v
            for v, weight in adj[u]:
                # new_mask includes all vertices from 'mask' plus vertex 'v'.
                # The '|' operator correctly adds 'v' to the visited set.
                new_mask = mask | (1 << v)
                new_cost = dp[mask][u] + weight

                # If this new path is shorter than the current shortest path to (new_mask, v),
                # update the dp table.
                if new_cost < dp[new_mask][v]:
                    dp[new_mask][v] = new_cost
    
    # After filling the dp table, we need to find the minimum cost for a walk
    # that visits all N vertices. This means the final mask must be (1 << N) - 1,
    # which represents all vertices from 0 to N-1 being visited.
    all_visited_mask = (1 << N) - 1
    min_total_weight = INF

    # Check all possible ending vertices for the walk that visited all vertices.
    for v in range(N):
        min_total_weight = min(min_total_weight, dp[all_visited_mask][v])

    # If min_total_weight is still INF, it means no walk exists that visits all vertices.
    if min_total_weight == INF:
        print("No")
    else:
        print(min_total_weight)

# Call the solve function to execute the program
solve()