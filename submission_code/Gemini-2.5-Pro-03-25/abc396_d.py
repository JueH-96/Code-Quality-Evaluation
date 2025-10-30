# YOUR CODE HERE
import sys

# Setting a higher recursion depth limit isn't strictly necessary for N=10
# as the maximum path length is N-1=9, well below the typical default limit of 1000.
# We can comment it out or leave it as a potential safeguard.
# sys.setrecursionlimit(2000) 

def solve():
    """
    Reads graph information from stdin, finds the minimum XOR sum path
    from vertex 1 to vertex N using DFS, and prints the result to stdout.
    """
    N, M = map(int, sys.stdin.readline().split())
    
    # Initialize adjacency list for all vertices 1 to N using a dictionary
    # where keys are vertex numbers and values are lists of tuples (neighbor, weight).
    # This ensures all vertices have an entry, even if they have degree 0 (which won't happen here due to connectivity).
    adj = {i: [] for i in range(1, N + 1)}
    
    # Read M edges and populate the adjacency list
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        # Graph is undirected, so add edge representation for both directions
        adj[u].append((v, w))
        adj[v].append((u, w))

    # Initialize minimum XOR sum found so far.
    # We need a value guaranteed to be larger than any possible XOR sum.
    # Edge weights w_i are non-negative and < 2^60. The XOR sum of multiple
    # such values could theoretically exceed 2^60 slightly if many high bits are set,
    # but it will still fit within standard 64-bit integer range (or Python's arbitrary precision integers).
    # (1 << 61) is a safe upper bound initial value. float('inf') could also work conceptually, 
    # but using large integer is cleaner for bitwise operations.
    min_xor_found = (1 << 61) 

    # Define the recursive Depth First Search function
    def dfs(u, visited_mask, current_xor_sum):
        """
        Performs DFS to find simple paths from vertex u to N.
        A simple path does not visit any vertex more than once.
        
        Args:
            u: The current vertex in the path.
            visited_mask: A bitmask where the i-th bit is set if vertex i+1 has been visited
                          in the current path. For example, vertex 1 corresponds to bit 0, vertex 2 to bit 1, etc.
            current_xor_sum: The cumulative XOR sum of edge weights along the path from the start vertex 1 to the current vertex u.
        """
        # Use 'nonlocal' keyword to indicate that we intend to modify the variable 'min_xor_found'
        # defined in the enclosing scope (the 'solve' function).
        nonlocal min_xor_found
        
        # Base case: If the current vertex u is the destination vertex N
        if u == N:
            # A path from 1 to N has been found.
            # Update the global minimum XOR sum if the XOR sum of this path is smaller.
            min_xor_found = min(min_xor_found, current_xor_sum)
            # Stop exploring further down this path (return to the caller).
            return 

        # Recursive step: Explore all neighbors of the current vertex u
        # adj[u] provides a list of tuples (neighbor_vertex, edge_weight)
        for v, w in adj[u]:
            # Check if the neighbor vertex 'v' has already been visited in the current path.
            # This is done by checking if the bit corresponding to 'v' is set in 'visited_mask'.
            # Vertex 'v' corresponds to the (v-1)-th bit.
            if not (visited_mask & (1 << (v-1))):
                # If vertex 'v' has not been visited:
                
                # Create the new visited mask for the recursive call by setting the bit for vertex v.
                # Use bitwise OR operation.
                new_visited_mask = visited_mask | (1 << (v-1))
                
                # Calculate the new XOR sum for the path extended to vertex v.
                # Use bitwise XOR operation.
                new_xor_sum = current_xor_sum ^ w
                
                # Make the recursive call to continue the DFS from neighbor v.
                # Pass the updated visited mask and XOR sum.
                dfs(v, new_visited_mask, new_xor_sum)
        
        # After exploring all neighbors, the function returns, effectively backtracking.

    # Start the DFS process from the source vertex 1.
    # Initial state:
    #   Current vertex is 1.
    #   Visited mask has only the 0-th bit set (representing vertex 1): 1 << 0 evaluates to 1.
    #   Initial XOR sum is 0, as no edges have been traversed yet.
    dfs(1, 1 << 0, 0) 
    
    # After the DFS has explored all possible simple paths from 1 to N,
    # print the minimum XOR sum found.
    # The problem constraints (connected graph, N >= 2) guarantee that at least one path exists,
    # so min_xor_found will hold the minimum XOR sum of such a path.
    print(min_xor_found)

# Execute the solve function to run the program
solve()