from collections import defaultdict
from typing import List

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        # Build the tree as an adjacency list: each entry will have (neighbor, weight)
        tree = defaultdict(list)
        n = len(nums)
        for u, v, w in edges:
            tree[u].append((v, w))
            tree[v].append((u, w))
        
        # Global variables to store the best result found:
        best_length = 0   # longest path length (sum of edge weights)
        best_nodes = 1    # for paths yielding best_length, the minimum number of nodes
        
        # Dictionary to store the most recent index (depth) of a value in the current DFS path.
        last_occ = {}
        # This list records the cumulative sum at each depth of the DFS path.
        prefix = []
        
        # DFS function.
        # Parameters:
        #   node: current node being visited.
        #   parent: parent of the current node (to avoid going back).
        #   depth: current depth along the DFS (which serves as an index into the DFS path).
        #   cum: current cumulative sum (total edge weight from the starting node in the DFS branch).
        #   window_start: the index in the DFS path from which the current unique window starts.
        def dfs(node: int, parent: int, depth: int, cum: int, window_start: int):
            nonlocal best_length, best_nodes
            
            # Before processing this node, adjust the window_start if a duplicate value exists.
            # If the current node's value already appeared in the DFS path at an index >= window_start,
            # then in order to keep the window unique, we must start the window from one index after the previous occurrence.
            prev_index = last_occ.get(nums[node], -1)
            if prev_index >= window_start:
                window_start = prev_index + 1
            
            # Remember the old value for backtracking.
            old_val = last_occ.get(nums[node], None)
            last_occ[nums[node]] = depth
            
            # Append the current cumulative sum to prefix.
            prefix.append(cum)
            # The length of the current valid (unique) special path that ends here
            # is the difference between the current cumulative sum and the cumulative sum at window_start.
            curr_length = cum - prefix[window_start]
            # The number of nodes in the subpath is (current index - window_start + 1).
            curr_nodes = depth - window_start + 1
            
            # Update global best if needed.
            if curr_length > best_length:
                best_length = curr_length
                best_nodes = curr_nodes
            elif curr_length == best_length and curr_nodes < best_nodes:
                best_nodes = curr_nodes
            
            # Continue DFS for children.
            for (nbr, w) in tree[node]:
                if nbr == parent:
                    continue
                dfs(nbr, node, depth + 1, cum + w, window_start)
            
            # Backtrack: remove the current node's cumulative sum.
            prefix.pop()
            # Revert the last occurrence entry.
            if last_occ.get(nums[node], None) == depth:
                if old_val is None:
                    del last_occ[nums[node]]
                else:
                    last_occ[nums[node]] = old_val
        
        # Start DFS from the root (node 0), with initial cumulative sum 0 and window_start 0.
        dfs(0, -1, 0, 0, 0)
        return [best_length, best_nodes]