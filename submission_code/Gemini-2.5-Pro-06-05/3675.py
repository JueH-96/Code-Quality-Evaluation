import sys
from typing import List

class Solution:
  def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
    n = len(edges) + 1
    if n <= 1:
        return 0
    
    # Increase recursion limit for deep trees to avoid RecursionError.
    # The default limit (often 1000) might not suffice for n=10^5.
    if n + 50 > sys.getrecursionlimit():
        sys.setrecursionlimit(n + 50) 
    
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
        
    def dfs(u: int, p: int) -> List[int]:
        """
        Performs a post-order traversal to compute optimal sums for the subtree at u.
        
        Returns:
            A list of two values:
            [max_sum_if_parent_edge_kept, max_sum_if_parent_edge_removed]
        """
        
        # This stores the potential "extra gain" from keeping an edge to a child.
        extra_gains = []
        
        # This is the sum of weights from subtrees of u's children,
        # assuming we cut all edges from u to its children.
        base_sum = 0
        
        for v, w in adj[u]:
            if v == p:
                continue
            
            # Post-order traversal: visit children first.
            child_sums = dfs(v, u)
            
            # child_sums[0] is max sum in v's subtree if (u,v) is kept.
            # child_sums[1] is max sum in v's subtree if (u,v) is removed.
            
            # If we remove edge (u, v), the max sum from v's subtree is child_sums[1].
            # We accumulate this into a base sum.
            base_sum += child_sums[1]
            
            # If we keep edge (u, v), the total contribution from this branch is (w + child_sums[0]).
            # The net gain of keeping (u, v) over removing it is (w + child_sums[0]) - child_sums[1].
            gain = w + child_sums[0] - child_sums[1]
            extra_gains.append(gain)
        
        # To maximize sum, we greedily pick the child edges with the highest positive extra gains.
        extra_gains.sort(reverse=True)
        
        # Case 1: Edge (u, parent) is NOT kept. Node u can connect to at most k children.
        sum_if_parent_edge_removed = base_sum
        for i in range(min(k, len(extra_gains))):
            if extra_gains[i] > 0:
                sum_if_parent_edge_removed += extra_gains[i]
            else:
                # No point in adding non-positive gains
                break

        # Case 2: Edge (u, parent) is KEPT. Node u's degree is already 1.
        # It can connect to at most k-1 more children.
        sum_if_parent_edge_kept = base_sum
        # The loop range handles k=1 correctly (range becomes empty).
        for i in range(min(k - 1, len(extra_gains))):
            if extra_gains[i] > 0:
                sum_if_parent_edge_kept += extra_gains[i]
            else:
                break
        
        return [sum_if_parent_edge_kept, sum_if_parent_edge_removed]

    # Start the traversal from an arbitrary root (e.g., node 0) with a dummy parent (-1).
    final_sums = dfs(0, -1)
    
    # The result for the whole tree is the one where the root has no parent edge.
    return final_sums[1]