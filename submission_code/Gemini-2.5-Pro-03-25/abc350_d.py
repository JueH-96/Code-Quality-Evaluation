# YOUR CODE HERE
import sys

# Increase recursion depth limit for safety with large N, although path compression makes it unlikely to be hit.
# Standard Python limit is ~1000, N can be up to 2*10^5.
# Setting it high ensures that even worst-case (though unlikely with optimizations) scenarios are covered.
# The sys.setrecursionlimit call might not be necessary depending on the system/Python version, 
# but it's a safe precaution for competitive programming problems with large inputs.
sys.setrecursionlimit(2 * 10**5 + 10) 

def find(parent, i):
    """
    Finds the root of the set containing element i using path compression.
    
    Args:
        parent: A list where parent[i] is the parent of element i.
        i: The element whose set root needs to be found.
        
    Returns:
        The root of the set containing element i.
    """
    # Base case: if i is its own parent, it's the root.
    if parent[i] == i:
        return i
    # Recursive call to find the root of the parent.
    # Path compression: Update parent[i] to point directly to the root.
    # This flattens the tree structure for future find operations on elements in this path.
    parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, size, i, j):
    """
    Unites the sets containing elements i and j using the union by size heuristic.
    
    Args:
        parent: A list where parent[i] is the parent of element i.
        size: A list where size[root] stores the size of the set rooted at 'root'.
        i, j: The elements whose sets are to be united.
        
    Returns:
        True if the sets were different and successfully united, False otherwise.
    """
    # Find the roots of the sets containing i and j.
    root_i = find(parent, i)
    root_j = find(parent, j)
    
    # If i and j are already in the same set (same root), do nothing.
    if root_i != root_j:
        # Union by size: Attach the root of the smaller set to the root of the larger set.
        # This heuristic helps keep the tree depth relatively shallow, improving efficiency.
        if size[root_i] < size[root_j]:
            parent[root_i] = root_j
            # Update the size of the resulting set (whose root is root_j).
            size[root_j] += size[root_i]
        else: # size[root_i] >= size[root_j]
            parent[root_j] = root_i
            # Update the size of the resulting set (whose root is root_i).
            size[root_i] += size[root_j]
        # Return True to indicate that a merge operation was performed.
        return True 
    # Return False if i and j were already in the same set.
    return False

def solve():
    """
    Reads input, solves the problem, and prints the answer.
    The problem asks for the maximum number of 'friend-of-a-friend' operations possible.
    This equals the total edges needed to make each connected component a clique, minus initial edges.
    """
    # Read number of users N and initial friendships M from standard input.
    N, M = map(int, sys.stdin.readline().split())
    
    # Initialize Disjoint Set Union (DSU) data structure.
    # We use 1-based indexing for users (1 to N), so arrays are size N+1. Index 0 is unused.
    parent = list(range(N + 1)) # parent[i] stores the parent of node i. Initially, each node is its own parent.
    size = [1] * (N + 1)      # size[i] stores the size of the component rooted at i (only valid for roots). Initially, each component has size 1.

    # Keep track of the number of initial edges provided in the input.
    initial_edges = M

    # Process all M initial friendships to build the connected components using DSU.
    for _ in range(M):
        # Read a friendship pair (u, v).
        u, v = map(int, sys.stdin.readline().split())
        # Combine the sets containing users u and v. If they are already in the same set, union does nothing.
        union(parent, size, u, v)
            
    # After processing all friendships, the DSU structure represents the connected components of the graph.
    # Each set in the DSU corresponds to a connected component.
    # The core idea is that the operation can be performed until each connected component becomes a complete graph (clique).
    # We need to calculate the total number of edges required for this final state.
    total_final_edges = 0
    
    # Use a set to keep track of the roots of components already processed. This ensures we sum edges for each component only once.
    roots_processed = set()
    
    # Iterate through each user from 1 to N to identify their component and calculate potential edges.
    for i in range(1, N + 1):
        # Find the representative (root) of the component user i belongs to.
        # The find operation also performs path compression, making subsequent finds faster.
        root = find(parent, i) 
        
        # Check if we have already calculated the clique edges for the component identified by this root.
        if root not in roots_processed:
            # If this component's root hasn't been processed yet:
            # Get the size of this component (k). The size is stored in the size array at the root index.
            k = size[root] 
            
            # Calculate the number of edges required for a clique of size k. The formula is k * (k - 1) / 2.
            # Use integer division // to ensure the result is an integer. Python's integers have arbitrary precision.
            clique_edges = k * (k - 1) // 2 
            
            # Add this component's potential clique edges to the total count across all components.
            total_final_edges += clique_edges
            
            # Mark this component's root as processed so we don't count its contribution again.
            roots_processed.add(root)
            
    # The maximum number of operations possible is the difference between the total edges in the final clique state
    # and the number of edges that were already present initially.
    max_operations = total_final_edges - initial_edges
    
    # Print the final answer to standard output.
    print(max_operations)

# Run the solve function to execute the main logic of the program when the script is run.
solve()