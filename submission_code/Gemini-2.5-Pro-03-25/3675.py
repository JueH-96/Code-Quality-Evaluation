import sys
from typing import List, Dict, Tuple

# Attempt to set a higher recursion depth needed for deep trees up to N=10^5.
# This is necessary because the depth of the recursion could potentially reach N
# in the case of a path graph or other deep tree structures.
# The default recursion limit in Python (often 1000) might be insufficient.
# Note: Changing the recursion limit might not be allowed or effective in all environments
# (e.g., certain online judges or restricted execution environments).
try:
    # Set recursion depth based on max N constraint = 10^5. Add a small buffer.
    limit = 10**5 + 50
    # sys.setrecursionlimit(limit) 
    # Avoid potentially crashing on platforms where this is disallowed / problematic.
    # Check if necessary by comparing with current limit? Or assume it's needed?
    # For common competitive programming platforms, this is often increased already.
    # If this fails on a platform, an iterative solution (using an explicit stack) might be required.
    # Let's set it defensively. If it fails, it will raise an exception.
    current_limit = sys.getrecursionlimit()
    if current_limit < limit:
         sys.setrecursionlimit(limit)
except Exception as e:
    # If setting recursion limit fails, print a warning to stderr.
    # The program might fail with a RecursionError for deep tree inputs.
    print(f"Warning: Could not set recursion depth limit: {e}", file=sys.stderr)

class Solution:
  
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        """
        Calculates the maximum possible sum of weights of remaining edges after removing
        zero or more edges such that each node has degree at most k in the resulting graph.
        The input is guaranteed to be a tree.
        Uses dynamic programming on the tree structure.
        
        Args:
          edges: A list of edges, where each edge is [u, v, w] representing an edge
                 between node u and node v with weight w.
          k: The maximum allowed degree for any node in the resulting graph.
        
        Returns:
          The maximum possible sum of weights of the remaining edges.
        """
        
        # Number of nodes in the tree
        n = len(edges) + 1
        # Base case: constraints state 2 <= n. If n < 2 somehow, return 0.
        if n < 2: return 0 

        # Build adjacency list representation of the tree.
        # adj[u] contains a list of tuples (v, w) for neighbors v of u with edge weight w.
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))

        # Memoization table to store results of DP states to avoid recomputation.
        # Key: tuple (node u, parent p) representing the state in DFS traversal.
        # Value: list [dp_u_0, dp_u_1] storing computed max weights for the subtree rooted at u.
        memo: Dict[Tuple[int, int], List[float]] = {} 

        def dfs(u: int, p: int) -> List[float]:
            """
            Recursive helper function performing Depth First Search for Dynamic Programming.
            Computes the maximum weights for the subtree rooted at node u.
            
            Args:
              u: The current node being processed.
              p: The parent of node u in the DFS traversal (-1 indicates u is the root).
              
            Returns:
              A list [dp_u_0, dp_u_1] where:
              dp_u_0: max weight sum in subtree rooted at u, assuming edge (p, u) is NOT kept.
              dp_u_1: max weight sum in subtree rooted at u, assuming edge (p, u) IS kept. 
                      This value is -infinity if this state is invalid (e.g., if k=0).
            """
            
            # Check memoization table first
            state = (u, p)
            if state in memo:
                return memo[state]

            # Initialize base value: sum of max weights from children subtrees assuming edges (u, child) are NOT kept.
            base_value = 0.0
            # List to store the potential gains obtained by keeping edges to children.
            child_gains = [] 

            # Iterate over neighbors of u
            for v, w in adj[u]:
                # Skip the edge leading back to the parent node p
                if v == p: 
                    continue
                
                # Recursively call dfs for child v
                dp_v = dfs(v, u) # Returns [dp_v_0, dp_v_1] for child v
                
                # Add the max weight from child v's subtree assuming edge (u,v) is not kept
                base_value += dp_v[0] 
                
                # Calculate the potential gain of keeping the edge (u,v).
                # Gain = (Weight(u,v) + DP value of v if (u,v) kept) - (DP value of v if (u,v) not kept)
                # Initialize gain to -infinity, indicating keeping edge is initially considered impossible/invalid.
                current_gain = -float('inf') 
                # Check if the state dp_v[1] (child v keeps edge to parent u) is valid.
                # dp_v[1] is invalid (-inf) if k=0, or if keeping edge (u,v) would violate constraints at v.
                if dp_v[1] != -float('inf'):
                    # If dp_v[1] is valid, calculate the actual gain.
                    current_gain = (w + dp_v[1]) - dp_v[0]
                
                # Store the calculated gain for this child edge.
                child_gains.append(current_gain)

            # Sort the gains in descending order. This allows greedily picking the best edges first.
            child_gains.sort(reverse=True)

            # Calculate dp[u][0]: Max weight sum if the edge (p, u) connecting to the parent is NOT kept.
            # In this case, node u can have up to k edges connecting to its children.
            dp_u_0 = base_value
            kept_count_0 = 0
            for gain in child_gains:
                # Only consider keeping edges if the gain is positive.
                # We can keep at most k such edges.
                if kept_count_0 < k and gain > 0: 
                    dp_u_0 += gain
                    kept_count_0 += 1
                # If gain is non-positive, no further edges should be kept (due to sorting).
                # Also stop if we have already selected k edges.
                elif gain <= 0: 
                     break
                # If kept_count_0 == k, the loop condition kept_count_0 < k fails for the next iteration.

            # Calculate dp[u][1]: Max weight sum if the edge (p, u) connecting to the parent IS kept.
            # This state requires that node u uses one degree capacity for the parent edge.
            # Thus, it can have at most k-1 edges connecting to its children.
            # This state is only possible if k >= 1.
            dp_u_1 = -float('inf') # Initialize as invalid state.
            if k >= 1: # Only proceed if k allows at least one edge (the parent edge).
                dp_u_1 = base_value
                kept_count_1 = 0
                for gain in child_gains:
                    # Only consider keeping edges if the gain is positive.
                    # We can keep at most k-1 such edges.
                    if kept_count_1 < k - 1 and gain > 0: 
                        dp_u_1 += gain
                        kept_count_1 += 1
                    # Stop if gain is non-positive or k-1 edges are already selected.
                    elif gain <= 0: 
                         break
                    # If kept_count_1 == k-1, the loop condition kept_count_1 < k - 1 fails for the next iteration.
            
            # Store the computed DP values in the memoization table.
            result = [dp_u_0, dp_u_1]
            memo[state] = result
            return result

        # Initiate DFS from an arbitrary root node (e.g., node 0) with parent -1.
        final_result_list = dfs(0, -1)
        # The final answer is the maximum weight sum for the entire tree. This corresponds to
        # the state where the root node (node 0) does not keep an edge to its (non-existent) parent.
        # This is captured by dp[0][0].
        final_max_weight = final_result_list[0]
        
        # The problem involves integer weights, so the maximum sum should be an integer.
        # Convert the result (which might be float due to using -inf) to integer.
        # Using int() is safe because intermediate sums of integers are exact within float precision limits.
        return int(final_max_weight)