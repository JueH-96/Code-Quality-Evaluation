# YOUR CODE HERE
import sys

# Increase recursion depth limit for deep trees
# Using a large enough value for N up to 2*10^5
try:
    # Set recursion depth slightly larger than max N to potentially avoid issues on edge cases
    # Standard competitive programming practice for potentially deep recursion on trees
    sys.setrecursionlimit(200000 + 50) 
except OverflowError: # Catch error if system limit is lower
     # Fallback to a large value if setting specific limit fails
     # This might depend on OS and Python installation configuration
     sys.setrecursionlimit(200000) 


def solve():
    N = int(sys.stdin.readline())
    
    # Base case: A tree with less than 5 nodes cannot form an alkane.
    # An alkane must have at least one node of degree 4, and by the formula
    # |V| = 3|V4| + 2 derived from degree sum properties of trees, 
    # the minimum size requires |V4|=1, so |V| = 3*1 + 2 = 5.
    if N < 5:
        print("-1")
        # Consume remaining input lines if N > 1 to avoid issues with potential subsequent test cases in some platforms
        for _ in range(N - 1):
             sys.stdin.readline()
        return

    adj = [[] for _ in range(N)]
    degree = [0] * N
    
    has_deg_ge_4 = False # Flag to track if any vertex has degree >= 4 in the original tree T
    
    # Read edges and build adjacency list, compute degrees
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        u -= 1 # Adjust to 0-based indexing
        v -= 1 # Adjust to 0-based indexing
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1
        
    # Check prerequisite: at least one vertex must have degree 4 or more in T
    # to potentially become degree 4 in the subgraph S. This is necessary for an alkane.
    for i in range(N):
        if degree[i] >= 4:
            has_deg_ge_4 = True
            break
    
    # If no vertex has degree >= 4, it's impossible to form an alkane.
    if not has_deg_ge_4:
        print("-1")
        return

    # Memoization dictionary for DP states to avoid recomputing for the same (node, parent) pair
    memo = {}

    # Use a list for max_alkane_size to allow modification within nested function scope (pass by reference effectively)
    max_alkane_size = [0] 

    # Depth First Search function implementing the dynamic programming logic
    def dfs(u, p):
        """
        Computes DP states for node u, assuming p is its parent in DFS traversal.
        Returns a tuple representing the DP states for node u:
        (dp0, dp1, dp2, dp3, dp4)
        Definitions:
        dp0: max size of an alkane fully contained within the subtree T_u rooted at u, NOT containing u itself.
        dp1: max size of an alkane fragment rooted at u containing u, where u has degree 1 IN THE FRAGMENT, and needs connection UPWARDS to its parent p. This represents u possibly being a leaf connected to parent. Size is 1 if the fragment only contains node u.
        dp2: max size of an alkane fragment rooted at u containing u, where u has degree 3 IN THE FRAGMENT, and needs connection UPWARDS to its parent p (to potentially become degree 4). Requires degree[u] >= 4 in the original tree T. Represents u possibly being an internal node connected to parent.
        dp3: max size of a completed alkane rooted at u containing u, where u has degree 1 in the alkane, fully contained within T_u. Connects downwards to one child.
        dp4: max size of a completed alkane rooted at u containing u, where u has degree 4 in the alkane, fully contained within T_u. Connects downwards to four children. Requires degree[u] >= 4 in T.
        """
        
        state_key = (u, p) # Use (node, parent) tuple as key for memoization
        if state_key in memo:
           return memo[state_key]

        # Collect results from children recursive calls
        children_results = []
        for v in adj[u]:
            if v != p: # Avoid going back to parent
                 children_results.append(dfs(v, u))

        # Calculate dp values for node u based on children results

        # Compute dp0: Max size alkane fully contained within T_u, not including u.
        # This is the max over completed alkanes found in any child's subtree.
        current_dp0 = 0
        for res_v in children_results:
            # An alkane in T_v could be fully contained within T_v (res_v[0]), 
            # or rooted at v as leaf (res_v[3]), or rooted at v as internal (res_v[4]).
            current_dp0 = max(current_dp0, res_v[0], res_v[3], res_v[4])
        
        # Compute contributions from children connecting upwards.
        # A child v can connect upwards if it can form a fragment ending in v
        # where v needs connection upwards (state dp1 or dp2).
        child_contributions = []
        for res_v in children_results:
             # C_v = max contribution from child v if u connects to it.
             # A child v can connect up either as a leaf (state 1) or as an internal node needing one more connection (state 2).
             C_v = max(res_v[1], res_v[2]) 
             # Only consider positive contributions (valid fragments). A contribution of 0 means the child cannot connect upwards in a useful way.
             if C_v > 0: 
                child_contributions.append(C_v)
        
        # Sort contributions descendingly to easily pick top K
        child_contributions.sort(reverse=True)
        num_children_contrib = len(child_contributions)

        # Compute dp1: u connects upwards as a leaf. Fragment size is 1 (just u).
        # Always possible for any node u.
        current_dp1 = 1 

        # Compute dp2: u connects upwards as internal (needs 1 more edge to become degree 4). 
        # Requires degree[u] >= 4 in T. Needs 3 children connections downwards.
        current_dp2 = 0 # Default to 0 (invalid/impossible state for this u)
        if degree[u] >= 4 and num_children_contrib >= 3:
             # Size = 1 (for u) + sum of contributions from top 3 children
             current_dp2 = 1 + sum(child_contributions[0:3])

        # Compute dp3: u is root of a completed alkane fragment, u has degree 1. Needs 1 child connection downwards.
        current_dp3 = 0 # Default to 0
        if num_children_contrib >= 1:
            # Size = 1 (for u) + contribution from the best child (largest contribution)
            current_dp3 = 1 + child_contributions[0]

        # Compute dp4: u is root of a completed alkane, u has degree 4. Needs degree[u] >= 4 in T. Needs 4 children connections downwards.
        current_dp4 = 0 # Default to 0
        if degree[u] >= 4 and num_children_contrib >= 4:
             # Size = 1 (for u) + sum of contributions from top 4 children
             current_dp4 = 1 + sum(child_contributions[0:4])

        # Update the overall maximum alkane size found so far across the entire tree.
        # Compare current max with completed alkanes found: 
        # Those fully within T_u (represented by dp0) and those rooted at u (dp3, dp4).
        max_alkane_size[0] = max(max_alkane_size[0], current_dp0, current_dp3, current_dp4)

        # Store result in memoization table and return it
        result = (current_dp0, current_dp1, current_dp2, current_dp3, current_dp4)
        memo[state_key] = result
        return result

    # Start DFS from an arbitrary root, node 0. Parent is -1 (non-existent node).
    # The choice of root does not affect the final max_alkane_size result due to tree structure and DP logic.
    dfs(0, -1)

    # Final check: The maximum size found must be at least 5 for a valid alkane.
    if max_alkane_size[0] >= 5:
        print(max_alkane_size[0])
    else:
        # If max size < 5, no valid alkane meeting the definition was found.
         print("-1")

# Execute the solve function to run the program
solve()