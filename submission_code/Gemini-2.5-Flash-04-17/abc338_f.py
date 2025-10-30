import sys

# Use 0-based indexing for vertices internally
# Vertices 1 to N map to 0 to N-1

def solve():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())

    # Adjacency list for graph: store tuples (neighbor, weight)
    # Use 0-based indexing
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        # Adjust to 0-based indexing
        u -= 1
        v -= 1
        adj[u].append((v, w))

    # DP state: dp[mask][u] = minimum cost of a walk ending at vertex u,
    # having visited all vertices in the set 'mask' (represented by bitmask) at least once.
    # mask is a bitmask where the i-th bit is set if vertex i (0-based) has been visited.
    INF = float('inf') # Use float('inf') for unreachable states
    dp = [[INF] * N for _ in range(1 << N)]

    # Base cases: A walk starting and ending at vertex i, having visited only {i}, costs 0.
    # We can start at any vertex.
    for i in range(N):
        dp[1 << i][i] = 0

    # Bellman-Ford like relaxation on the state space (mask, u)
    # A state is (mask, u), representing having visited all vertices in 'mask' and currently at 'u'.
    # A transition occurs along an edge (u, v) with weight w from the original graph.
    # The transition is from state (mask, u) to state (mask | (1 << v), v) with cost increase w.
    # The state space has N * 2^N vertices.
    # The number of edges in the state graph is sum over all states (mask, u) of degree+(u) in original graph.
    # This sum is upper bounded by N * 2^N * M.
    # Since there are no negative cycles in the original graph, there are no negative cycles
    # in the state graph where edges correspond to single steps in the original graph
    # and state cost is cumulative sum of edge weights.
    # Standard Bellman-Ford requires |V_state| - 1 iterations, which is too large here.
    # However, the mask component of the state is non-decreasing.
    # The number of iterations required is related to the maximum useful path length in the state graph.
    # A walk must visit N distinct vertices. This takes at least N-1 edges if they are all new vertices.
    # After all vertices are visited, additional edges can be traversed to potentially reduce cost
    # due to negative edge weights (without forming negative cycles).
    # The number of iterations needed is likely related to N. A safe upper bound often used for this type of DP
    # with negative weights (but no negative cycles) is N * N or 2 * N.
    # Let's use 2 * N iterations. This should be sufficient for information to propagate through
    # states where the mask increases up to N times and then potentially traverse
    # within the fully visited set to optimize.

    # Number of Bellman-Ford iterations. 2*N is a common safe bound in such problems.
    num_bf_iterations = 2 * N

    for _iter in range(num_bf_iterations):
        # In each iteration, we iterate through all possible state transitions (mask, u) -> (next_mask, v).
        # Iterate through all possible masks
        for mask in range(1 << N):
            # Iterate through all possible current vertices u
            for u in range(N):
                # If state (mask, u) is currently unreachable or not yet found, skip
                if dp[mask][u] == INF:
                    continue

                # Explore outgoing edges from u in the original graph
                for v, w in adj[u]:
                    # The next state is (mask | (1 << v), v)
                    next_mask = mask | (1 << v)

                    # Relax the state (next_mask, v)
                    if dp[mask][u] + w < dp[next_mask][v]:
                         dp[next_mask][v] = dp[mask][u] + w

    # After the Bellman-Ford iterations, the minimum cost to visit all vertices
    # is the minimum value among all states (final_mask, u), where final_mask
    # represents the set of all N vertices being visited.
    final_mask = (1 << N) - 1
    min_cost = INF

    # Find the minimum cost among all possible ending vertices u after visiting all vertices
    for u in range(N):
        min_cost = min(min_cost, dp[final_mask][u])

    # If the minimum cost is still INF, it means no walk exists that visits all vertices.
    if min_cost == INF:
        print("No")
    else:
        print(min_cost)

solve()