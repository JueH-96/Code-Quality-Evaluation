import sys

# Increase recursion depth limit if needed. N=10 means max path length 9.
# Default limit is usually 1000. Setting it to 2000 is safe.
sys.setrecursionlimit(2000)

# Use a large value for infinity.
# Maximum possible value of an edge weight is less than 2^60.
# The XOR sum of any path will also be less than 2^60.
# Let's use a constant equal to 2^60, which is strictly greater than any possible path XOR sum.
INF = 1 << 60

def solve():
    # Read N (number of vertices) and M (number of edges)
    N, M = map(int, sys.stdin.readline().split())

    # Build the graph using an adjacency list.
    # graph[u] will store a list of tuples (neighbor_v, edge_weight_w).
    # Since the graph is undirected, add edges in both directions.
    graph = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        # Add edge (u, v) with weight w
        graph[u].append((v, w))
        # Add edge (v, u) with weight w
        graph[v].append((u, w))

    # Variable to store the minimum XOR sum found across all simple paths from vertex 1 to vertex N.
    # Initialize it with a value greater than any possible path XOR sum.
    min_xor_sum = INF

    # Array to keep track of visited vertices during the current simple path exploration.
    # This prevents visiting the same vertex twice within a single path, ensuring simplicity.
    # Size N+1 for 1-based indexing.
    visited = [False] * (N + 1)

    # Recursive function to find simple paths from the current vertex u to the destination vertex N.
    # current_xor is the XOR sum of edge weights accumulated along the path from the start (vertex 1) to u.
    def find_min_xor_path(u, current_xor):
        # Use 'nonlocal' keyword to be able to modify the min_xor_sum variable
        # defined in the enclosing 'solve' function's scope.
        nonlocal min_xor_sum

        # Mark the current vertex as visited for the current path being explored.
        visited[u] = True

        # Base case: If the current vertex u is the destination vertex N.
        if u == N:
            # We found a simple path from 1 to N.
            # Update the minimum XOR sum found so far if the current path's XOR sum is smaller.
            min_xor_sum = min(min_xor_sum, current_xor)
        else:
            # Recursive step: Explore the neighbors of the current vertex u.
            # Iterate through all adjacent vertices v and their edge weights w.
            for v, w in graph[u]:
                # Check if the neighbor vertex v has not been visited yet in the current path.
                # This is crucial for ensuring the path remains simple.
                if not visited[v]:
                    # If the neighbor is not visited, extend the current path to v.
                    # Recursively call the function for the neighbor v.
                    # The new current_xor for the next step is the current_xor XORed with the edge weight w.
                    find_min_xor_path(v, current_xor ^ w)

        # Backtrack: Before returning from the recursive call for vertex u, unmark it as visited.
        # This allows other simple paths (that do not include u at this specific point in the path from 1)
        # to potentially visit vertex u later. This is essential for exploring all distinct simple paths.
        visited[u] = False

    # Start the depth-first search process from the source vertex 1.
    # The initial XOR sum is 0 as we haven't traversed any edges yet.
    find_min_xor_path(1, 0)

    # After exploring all simple paths from 1 to N, print the minimum XOR sum found.
    print(min_xor_sum)

# Call the solve function to execute the program logic.
# When run by the test system, this function will read from stdin and write to stdout.
solve()