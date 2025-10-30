# YOUR CODE HERE
import sys

def solve():
    """
    Reads the graph, computes the number of reachable pairs, and prints the result.
    The graph is a functional graph, where each vertex has an out-degree of 1.
    Such a graph consists of one or more components, each containing exactly one cycle
    and some trees whose edges point towards the cycle vertices.

    The problem asks for the sum of the number of reachable vertices from each vertex u.
    Total = sum_{u in V} |{v in V | u can reach v}|

    This can be solved efficiently using an iterative, DFS-like approach.
    We use a `dp` array to store computed results and track node states:
    - dp[i] = 0: unvisited
    - dp[i] = -1: currently being visited (on the current traversal path)
    - dp[i] > 0: computation finished, value is the result.
    """
    try:
        N = int(sys.stdin.readline())
        # The problem has 1-based indexing for vertices, we convert to 0-based.
        a = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handles empty input at EOF in some environments.
        return

    next_node = [x - 1 for x in a]

    dp = [0] * N

    for i in range(N):
        # If dp[i] is already computed (>0), we skip.
        # A new traversal starts only from a completely unvisited node (dp[i] == 0).
        if dp[i] == 0:
            path = []
            curr = i
            # Trace the path until we hit a node that is not 'unvisited'.
            while dp[curr] == 0:
                dp[curr] = -1  # Mark as 'visiting'
                path.append(curr)
                curr = next_node[curr]

            # Case 1: A cycle is detected.
            # `curr` is on the current path, so dp[curr] is -1.
            if dp[curr] == -1:
                cycle_start_idx = path.index(curr)
                cycle_len = len(path) - cycle_start_idx
                
                # All nodes in the cycle can reach `cycle_len` nodes.
                for j in range(cycle_start_idx, len(path)):
                    dp[path[j]] = cycle_len
                
                # Process the tail leading to the cycle.
                for j in range(cycle_start_idx - 1, -1, -1):
                    dp[path[j]] = 1 + dp[path[j+1]]
            
            # Case 2: The path leads to a previously computed component.
            # dp[curr] will be > 0.
            else:
                count = dp[curr]
                for node in reversed(path):
                    count += 1
                    dp[node] = count
    
    print(sum(dp))

solve()