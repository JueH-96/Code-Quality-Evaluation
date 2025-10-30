# YOUR CODE HERE
import sys

# The default recursion limit in Python is typically 1000, which is
# more than sufficient for a maximum path length of N=10.
# No need to increase the recursion limit.

def main():
    """
    Reads graph data and computes the minimum XOR sum of a simple path
    from vertex 1 to N.
    """
    try:
        # Read graph dimensions
        N, M = map(int, input().split())
    except (IOError, ValueError):
        # Handle empty or malformed input.
        return

    # Adjacency list representation of the graph.
    # adj[i] stores a list of (neighbor, weight) tuples for vertex i+1.
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v, w = map(int, input().split())
        # Convert to 0-based indexing for array access
        u -= 1
        v -= 1
        adj[u].append((v, w))
        adj[v].append((u, w))

    # This list will store the XOR sums of all simple paths found.
    path_xors = []

    def dfs(u, current_xor, visited):
        """
        A recursive DFS function to find all simple paths and their XOR sums.
        
        Parameters:
        u (int): The current vertex (0-indexed).
        current_xor (int): The XOR sum of edge weights on the path from the start to u.
        visited (int): A bitmask representing the set of visited vertices on the current path.
        """
        # Base case: If we have reached the destination (vertex N, which is N-1).
        if u == N - 1:
            path_xors.append(current_xor)
            return

        # Explore all neighbors of the current vertex u.
        for v_neighbor, w_edge in adj[u]:
            # Check if the neighbor has already been visited on this path using the bitmask.
            if not ((visited >> v_neighbor) & 1):
                # If not visited, continue the search from this neighbor.
                # Update the visited mask by setting the bit for the new vertex.
                new_visited = visited | (1 << v_neighbor)
                # Update the XOR sum.
                new_xor = current_xor ^ w_edge
                dfs(v_neighbor, new_xor, new_visited)

    # Start the DFS from the source vertex (vertex 1, which is index 0).
    start_node = 0
    # The initial visited mask contains only the start node.
    initial_visited_mask = 1 << start_node
    # The initial path has an XOR sum of 0.
    dfs(start_node, 0, initial_visited_mask)

    # The problem constraints guarantee that a path from 1 to N exists.
    # Therefore, the path_xors list will not be empty.
    # Find and print the minimum XOR sum from all the paths found.
    print(min(path_xors))

if __name__ == "__main__":
    main()