import sys

def solve():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())

    # Initialize adjacency list for the graph
    # Vertices are 1-indexed in input, convert to 0-indexed for internal use
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        u -= 1  # Adjust to 0-indexed
        v -= 1  # Adjust to 0-indexed
        adj[u].append((v, w))
        adj[v].append((u, w)) # Graph is undirected

    # Initialize minimum XOR sum to a very large value
    min_xor_path_sum = float('inf')

    # DFS function to explore paths
    # current_vertex: The current vertex in the path (0-indexed)
    # current_xor_sum: The accumulated XOR sum of edge weights along the path
    # visited_mask: A bitmask to keep track of visited vertices in the current path.
    #               If the k-th bit is set, vertex k has been visited.
    def dfs(current_vertex, current_xor_sum, visited_mask):
        nonlocal min_xor_path_sum

        # Base case: If we reached the destination vertex (N-1)
        if current_vertex == N - 1:
            min_xor_path_sum = min(min_xor_path_sum, current_xor_sum)
            return

        # Explore all neighbors of the current vertex
        for neighbor, weight in adj[current_vertex]:
            # Check if the neighbor has NOT been visited in the current path
            # (visited_mask & (1 << neighbor)) checks if the neighbor's bit is set
            if not (visited_mask & (1 << neighbor)):
                # Mark the neighbor as visited in the new mask
                new_visited_mask = visited_mask | (1 << neighbor)
                # Recurse with the neighbor, updated XOR sum, and new visited mask
                dfs(neighbor, current_xor_sum ^ weight, new_visited_mask)

    # Start the DFS from vertex 0 (which corresponds to 1-indexed vertex 1)
    # Initial XOR sum is 0. Only vertex 0 is visited initially (1 << 0).
    dfs(0, 0, 1 << 0)

    # Print the result
    sys.stdout.write(str(min_xor_path_sum) + '
')

# Call the solve function to run the program
solve()