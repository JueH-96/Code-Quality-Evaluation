import sys
sys.setrecursionlimit(200000)
from typing import List

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        # last index of value along current root->u path
        last_idx = {}
        # prefix distances at each depth
        dists = []
        
        best_len = 0
        best_nodes = float('inf')
        
        def dfs(u: int, parent: int, curr_dist: int, depth: int, curr_start: int):
            nonlocal best_len, best_nodes
            # Append current dist at this depth
            dists.append(curr_dist)
            
            val = nums[u]
            prev = last_idx.get(val, -1)
            # update sliding window start
            new_start = max(curr_start, prev + 1)
            # record this value's last occurrence
            old = last_idx.get(val, None)
            last_idx[val] = depth
            
            # compute suffix path length and node count
            # suffix runs from index new_start .. depth
            seg_len = curr_dist - dists[new_start]
            seg_nodes = depth - new_start + 1
            # update answer
            if seg_len > best_len:
                best_len = seg_len
                best_nodes = seg_nodes
            elif seg_len == best_len and seg_nodes < best_nodes:
                best_nodes = seg_nodes
            
            # DFS children
            for v, w in adj[u]:
                if v == parent:
                    continue
                dfs(v, u, curr_dist + w, depth + 1, new_start)
            
            # backtrack
            if old is None:
                del last_idx[val]
            else:
                last_idx[val] = old
            dists.pop()
        
        # Start DFS from root=0
        dfs(0, -1, 0, 0, 0)
        return [best_len, best_nodes]