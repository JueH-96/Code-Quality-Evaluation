import sys
from typing import List

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        # Increase recursion limit to handle deep trees
        sys.setrecursionlimit(10**7)
        
        n = len(nums)
        # Build undirected adjacency list: (neighbor, edge_weight)
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        # Prepare for tracking last occurrences of values on the current root->node path
        max_val = max(nums)
        last_occ = [-1] * (max_val + 1)
        
        # prefix_weights[d] = sum of edge weights from root to the node at depth d
        prefix_weights = [0] * n
        
        # Global answer variables
        max_weight = 0
        min_nodes = n + 1  # we'll take min over this
        
        def dfs(cur: int, parent: int, depth: int, pw: int, l: int):
            nonlocal max_weight, min_nodes
            # Record the prefix weight at this depth
            prefix_weights[depth] = pw
            
            # Remember previous last occurrence of this value to backtrack later
            prev_last = last_occ[nums[cur]]
            # If this value appeared within the current window [l..depth-1], we must move l
            if prev_last >= l:
                new_l = prev_last + 1
            else:
                new_l = l
            
            # Mark this node's value as seen at 'depth'
            last_occ[nums[cur]] = depth
            
            # Compute the weight of the unique-suffix path [new_l..depth]
            curr_weight = pw - prefix_weights[new_l]
            curr_nodes = depth - new_l + 1
            
            # Update global best
            if curr_weight > max_weight:
                max_weight = curr_weight
                min_nodes = curr_nodes
            elif curr_weight == max_weight and curr_nodes < min_nodes:
                min_nodes = curr_nodes
            
            # DFS children
            for nei, w in adj[cur]:
                if nei == parent:
                    continue
                dfs(nei, cur, depth + 1, pw + w, new_l)
            
            # Backtrack the last_occ for this value
            last_occ[nums[cur]] = prev_last
        
        # Start DFS from root = 0
        dfs(0, -1, 0, 0, 0)
        
        return [max_weight, min_nodes]