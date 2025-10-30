import sys
from typing import List

class Solution:
  def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
    n = 0
    # Determine n. Problem states "n nodes numbered 0 to n - 1" and "edges of length n - 1".
    # So n = len(edges) + 1 is reliable.
    # Constraints: 2 <= n <= 10^5. So len(edges) >= 1.
    n = len(edges) + 1
    
    # Adjacency list representation of the tree
    # self.adj[i] stores a list of (neighbor, weight) pairs for node i
    self.adj = [[] for _ in range(n)]
    for u_node, v_node, w_edge in edges:
        self.adj[u_node].append((v_node, w_edge))
        self.adj[v_node].append((u_node, w_edge))

    # DP table initialization
    # self.dp[u][0]: max sum in subtree u, edge (parent(u), u) is NOT selected (or u is root)
    # self.dp[u][1]: max sum in subtree u, edge (parent(u), u) IS selected.
    #           The weight of (parent(u),u) is not included here; it's added by the parent.
    self.dp = [[0] * 2 for _ in range(n)] 
    
    # Store k as a member variable for easy access in DFS
    self.k = k
    
    # Adjust Python's recursion limit if necessary.
    # Max recursion depth can be N for a path graph. Default limit (e.g., 1000-3000) might be too low.
    # Add a small buffer for safety, e.g., N + 200.
    required_recursion_limit = n + 200 
    try:
        current_recursion_limit = sys.getrecursionlimit()
        if required_recursion_limit > current_recursion_limit:
            sys.setrecursionlimit(required_recursion_limit)
    except RuntimeError:
        # This might happen in restricted environments (e.g., some online judges).
        # If so, an iterative DFS approach would be more robust.
        # For LeetCode, usually setting recursion limit is allowed.
        pass # If it fails, hope the default limit is enough or platform has high limit.

    # Start DFS from node 0 (arbitrary root). Parent of root is marked as -1 (dummy).
    self._dfs_recursive(0, -1) 

    # The answer is dp[root][0], as the root has no parent edge to take.
    return self.dp[0][0]

  def _dfs_recursive(self, u: int, p: int) -> None:
    # This method computes self.dp[u][0] and self.dp[u][1] using values from children.
    # It performs a post-order traversal implicitly due to recursion.
    
    # base_sum_from_children: Accumulates contributions from children subtrees,
    # assuming initially that NO edge (u,v) connecting to child v is taken.
    base_sum_from_children = 0
    
    # gains: Stores potential "gains" from choosing to connect to a child v.
    # Gain = (value if (u,v) taken) - (value if (u,v) not taken)
    gains = [] 

    for v, w_uv in self.adj[u]: # Iterate over neighbors of u
        if v == p: # Don't traverse back to the parent node
            continue
        
        # Recursively call DFS for child v. Its parent is u.
        self._dfs_recursive(v, u) 
        
        # After the recursive call, self.dp[v][0] and self.dp[v][1] are computed.
        
        # Contribution from v's subtree if edge (u,v) is NOT selected:
        base_sum_from_children += self.dp[v][0]
        
        # Calculate the marginal gain of selecting edge (u,v):
        # Value if (u,v) taken: w_uv (weight of edge) + self.dp[v][1] (v connects to parent u).
        # Value if (u,v) not taken: self.dp[v][0] (v does not connect to parent u).
        # The difference is the gain.
        gain = w_uv + self.dp[v][1] - self.dp[v][0]
        
        # We only consider selecting (u,v) if it provides a positive gain over not selecting it.
        if gain > 0:
            gains.append(gain)
    
    # Sort positive gains in descending order to pick the most valuable ones.
    gains.sort(reverse=True)

    # Calculate self.dp[u][0]: u does NOT connect to its parent p (or u is root).
    # In this state, u can use up to self.k edges to connect to its children.
    current_sum_0 = base_sum_from_children
    # Add the top 'self.k' positive gains.
    for i in range(min(len(gains), self.k)):
        current_sum_0 += gains[i]
    self.dp[u][0] = current_sum_0

    # Calculate self.dp[u][1]: u DOES connect to its parent p.
    # In this state, u uses one edge capacity for its parent,
    # so it can use up to self.k - 1 edges for its children.
    # Note: self.k >= 1 (from problem constraints), so self.k - 1 >= 0.
    current_sum_1 = base_sum_from_children
    num_child_edges_for_dp1 = self.k - 1
    # Add the top 'self.k - 1' positive gains.
    for i in range(min(len(gains), num_child_edges_for_dp1)):
        current_sum_1 += gains[i]
    self.dp[u][1] = current_sum_1