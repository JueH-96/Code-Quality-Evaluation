from typing import List
import collections

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        # Build adjacency list
        n = len(nums)
        adj = collections.defaultdict(list)
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        # Map to track last seen depth of each value
        val_last = {}
        # path_dist[d] = cumulative distance from root to the node at depth d
        path_dist = [0]
        
        # Results
        maxWeight = 0
        minNodes = 1
        
        def dfs(u: int, parent: int, depth: int, curr_dist: int, L: int):
            nonlocal maxWeight, minNodes
            
            v_val = nums[u]
            old_last = val_last.get(v_val, None)
            old_L = L
            
            # If the value was seen in the current window, move L
            if old_last is not None and old_last >= L:
                L = old_last + 1
            
            # Update last seen position for this value
            val_last[v_val] = depth
            
            # Compute the weight and node count of the current valid path [L..depth]
            start_dist = path_dist[L] if L > 0 else 0
            weight = curr_dist - start_dist
            node_count = depth - L + 1
            
            # Update global answers
            if weight > maxWeight:
                maxWeight = weight
                minNodes = node_count
            elif weight == maxWeight and node_count < minNodes:
                minNodes = node_count
            
            # Recurse to children
            for v, w in adj[u]:
                if v == parent:
                    continue
                path_dist.append(curr_dist + w)
                dfs(v, u, depth + 1, curr_dist + w, L)
                path_dist.pop()
            
            # Restore val_last for backtracking
            if old_last is None:
                del val_last[v_val]
            else:
                val_last[v_val] = old_last
            # L is local, no need to restore
        
        # Start DFS from root node 0
        dfs(0, -1, 0, 0, 0)
        
        return [maxWeight, minNodes]