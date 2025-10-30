# YOUR CODE HERE
import sys

# Increase recursion depth limit for potentially deep trees
# N max is 10^5. A value like 2*10^5 is usually safe for competitive programming environments.
try:
    # Set recursion depth limit slightly larger than max N * 2
    sys.setrecursionlimit(200005) 
except Exception as e:
    # If setting the recursion limit fails (e.g., due to platform restrictions),
    # the program will continue with the default limit. This might lead to
    # RecursionError for deep trees on some platforms, but this try-except
    # block prevents the program from crashing immediately upon failing to set the limit.
    # print(f"Warning: Failed to set recursion depth limit: {e}", file=sys.stderr) # Optional warning
    pass 

# Define global variables to be accessed by DFS functions
# Using globals simplifies passing state between recursive calls, common in competitive programming.
adj = [] # Adjacency list representation of the tree
C = [] # Cost array, will use 1-based indexing to align with vertex numbers (index 0 unused)
subtree_sum = [] # Stores sum of costs C_i for nodes in the subtree rooted at each node i (based on initial DFS from root 1)
min_f = float('inf') # Stores the minimum f(v) value found so far, initialized to positive infinity
S_total = 0 # Stores the total sum of all costs C_i across all vertices

def combined_dfs1(u, p, depth):
    """
    First DFS pass (performs tasks similar to a post-order traversal).
    Computes the sum of costs C_i for nodes in the subtree rooted at u (subtree_sum[u]).
    Also calculates the initial value f(1) = sum_{i=1}^N (C_i * d(1, i)) by summing contributions from all subtrees.
    
    Args:
        u (int): Current vertex being visited (1 to N).
        p (int): Parent vertex of u in the DFS traversal (0 for the root).
        depth (int): Depth of vertex u from the root (vertex 1), which is equal to the distance d(1, u).
        
    Returns:
        int: The contribution to f(1) from the subtree rooted at u. This is calculated as
             sum_{i in subtree rooted at u} (C[i] * d(1, i)).
             
    Side Effects:
        Updates the global `subtree_sum` array element `subtree_sum[u]` with the total cost sum
        of the subtree rooted at u.
    """
    global subtree_sum, C, adj
    
    # Initialize the subtree sum for node u with its own cost C[u].
    current_node_subtree_sum = C[u]
    # Initialize the contribution to f(1) from node u itself.
    current_node_f1_contrib = C[u] * depth
    
    # Explore neighbors of u
    for v in adj[u]:
        if v != p: # Ensure we don't traverse back up to the parent
            # Recursively call for child v. The depth increases by 1.
            child_f1_contrib = combined_dfs1(v, u, depth + 1)
            
            # After the recursive call returns, subtree_sum[v] contains the sum of costs
            # in the subtree rooted at v. Add this to the current node's subtree sum.
            current_node_subtree_sum += subtree_sum[v]
            # Add the contribution to f(1) from the child's subtree.
            current_node_f1_contrib += child_f1_contrib
            
    # Store the computed total subtree sum for u in the global array.
    subtree_sum[u] = current_node_subtree_sum
    # Return the total contribution to f(1) from the subtree rooted at u.
    return current_node_f1_contrib


def dfs_compute_f(u, p, current_f):
    """
    Second DFS pass (performs tasks similar to a pre-order traversal).
    Computes f(v) for all vertices v using the previously computed f(u) value (where u is parent of v)
    and the derived recurrence relation (rerooting technique). Updates the global minimum f value found.
    
    Args:
        u (int): Current vertex being visited (1 to N).
        p (int): Parent vertex of u in the DFS traversal (0 for the root).
        current_f (int): The value f(u), computed based on its parent's f value (or initialized for the root).
        
    Side Effects:
        Updates the global `min_f` variable if `current_f` (which represents f(u)) is smaller
        than the current minimum.
    """
    global min_f, S_total, subtree_sum, adj
    
    # Update the global minimum f value found so far.
    min_f = min(min_f, current_f)
    
    # Explore neighbors of u
    for v in adj[u]:
        if v != p: # Ensure we don't traverse back up to the parent
            # Compute f(v) using the rerooting formula: f(v) = f(u) + S_total - 2 * subtree_sum[v]
            # S_total is the total sum of costs over all nodes (sum C_i for all i).
            # subtree_sum[v] is the sum of costs in the subtree originally rooted at v (when vertex 1 was the root).
            # This formula holds when moving from a parent `u` to a child `v`.
            f_v = current_f + S_total - 2 * subtree_sum[v]
            
            # Recursively call for child v with its computed f value f_v.
            dfs_compute_f(v, u, f_v)

def solve():
    """
    Main function to handle input reading, orchestrate the two DFS passes required by the algorithm,
    and print the final minimum f value found.
    """
    global adj, C, subtree_sum, min_f, S_total

    # Read the number of vertices N
    N = int(sys.stdin.readline())
    
    # Initialize adjacency list (size N+1 for 1-based indexing)
    adj = [[] for _ in range(N + 1)]
    # Read N-1 edges and build the adjacency list representation of the tree
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # Read the costs C_1, ..., C_N
    C_list = list(map(int, sys.stdin.readline().split()))
    # Populate the 1-based indexed cost array C (index 0 is unused)
    C = [0] * (N + 1) 
    for i in range(N):
      C[i+1] = C_list[i]

    # Initialize the subtree_sum array (size N+1 for 1-based indexing)
    subtree_sum = [0] * (N + 1)

    # Perform the first DFS starting from vertex 1 (chosen as an arbitrary root)
    # This computes f(1) and fills the subtree_sum array for all nodes.
    f1 = combined_dfs1(1, 0, 0) # Vertex 1 is root, depth 0, its parent is dummy 0
    
    # The total sum of costs S_total is the subtree sum of the root (vertex 1), 
    # as its subtree contains all nodes in the tree.
    S_total = subtree_sum[1] 

    # Initialize the minimum f value found so far with f(1)
    min_f = f1

    # Perform the second DFS starting from vertex 1
    # This computes f(v) for all vertices v based on parent's f value and updates min_f.
    dfs_compute_f(1, 0, f1) # Start DFS with root 1 and its computed f value f1

    # Print the overall minimum f value found across all vertices
    print(min_f)

# Standard Python construct to run the main logic when the script is executed
if __name__ == '__main__':
    solve()