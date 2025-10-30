from typing import List
from collections import defaultdict, deque

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)
        # build adjacency list with edge length
        g = [[] for _ in range(n)]
        for u,v,w in edges:
            g[u].append((v,w))
            g[v].append((u,w))
            
        # distance from root (0) to each node (sum of edge lengths)
        dist = [0]*n
        parent = [-1]*n
        # iterative stack to avoid recursion depth issues (n<=5e4 so recursion ok but let's be safe)
        stack = [(0, -1)]
        order = []
        while stack:
            node, par = stack.pop()
            parent[node] = par
            order.append(node)
            for nei,w in g[node]:
                if nei == par: 
                    continue
                dist[nei] = dist[node] + w
                stack.append((nei,node))
        
        # data structures for DFS along root-path
        path_nodes = []         # node indices along current path
        path_dists = []         # dist from root for each node in path_nodes
        last_idx = {}           # value -> position in path_nodes where it last appeared
        
        max_len = 0
        min_nodes = 1
        
        # recursive DFS (using parent relation ensures we always move downward from root)
        def dfs(node, window_start):
            nonlocal max_len, min_nodes
            depth = len(path_nodes)           # index where 'node' will be pushed
            val = nums[node]
            
            prev_pos = last_idx.get(val, -1)  # -1 if not present
            start_idx = max(window_start, prev_pos + 1)
            
            # push current node
            path_nodes.append(node)
            path_dists.append(dist[node])
            last_idx[val] = depth
            
            # compute path statistics (unique path is [start_idx .. depth])
            curr_len = dist[node] - path_dists[start_idx]
            nodes_cnt = depth - start_idx + 1
            if curr_len > max_len:
                max_len = curr_len
                min_nodes = nodes_cnt
            elif curr_len == max_len:
                min_nodes = min(min_nodes, nodes_cnt)
            
            # traverse children
            for nei,w in g[node]:
                if nei == parent[node]:
                    continue
                dfs(nei, start_idx)
            
            # backtrack
            path_nodes.pop()
            path_dists.pop()
            if prev_pos == -1:
                del last_idx[val]
            else:
                last_idx[val] = prev_pos
        
        dfs(0,0)
        return [max_len, min_nodes]