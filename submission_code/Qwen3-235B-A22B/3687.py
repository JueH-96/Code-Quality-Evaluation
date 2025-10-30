from typing import List
from collections import deque

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)
        if n == 0:
            return [0, 0]
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        # Build children and edge_lengths using BFS
        parent = [-1] * n
        children = [[] for _ in range(n)]
        edge_lengths = [0] * n
        visited = [False] * n
        q = deque([0])
        visited[0] = True
        
        while q:
            u = q.popleft()
            for v, w in adj[u]:
                if not visited[v] and v != parent[u]:
                    parent[v] = u
                    visited[v] = True
                    children[u].append(v)
                    edge_lengths[v] = w
                    q.append(v)
        
        # Variables to track the maximum length and minimum nodes
        max_length = 0
        min_nodes = float('inf')
        
        # DFS traversal with backtracking
        prefix_sum = [0]  # prefix_sum[0] = 0, sum up to node 0 (root)
        current_start = 0
        last_occurrence = {}
        
        def dfs(node):
            nonlocal current_start, max_length, min_nodes
            val = nums[node]
            
            # Append edge length to prefix_sum if not root
            appended = False
            if node != 0:
                prefix_sum.append(prefix_sum[-1] + edge_lengths[node])
                appended = True
            
            # Check old index of val
            old_index = last_occurrence.get(val, -1)
            prev_start = current_start
            
            # Update current_start if there's a duplicate in the window
            if old_index >= current_start:
                current_start = old_index + 1
            
            # Update last_occurrence with current index
            last_occurrence[val] = len(prefix_sum) - 1
            
            # Calculate current path's length and node count
            current_sum = prefix_sum[-1]
            start_idx = current_start
            if start_idx < 0:
                start_idx = 0
            if start_idx >= len(prefix_sum):
                current_length_val = 0
            else:
                current_length_val = current_sum - prefix_sum[start_idx]
            
            current_nodes_val = len(prefix_sum) - current_start
            
            # Update global max and min
            if current_length_val > max_length:
                max_length = current_length_val
                min_nodes = current_nodes_val
            elif current_length_val == max_length:
                if current_nodes_val < min_nodes:
                    min_nodes = current_nodes_val
            
            # Recurse on children
            for child in children[node]:
                dfs(child)
            
            # Backtrack
            if appended:
                prefix_sum.pop()
            if old_index == -1:
                del last_occurrence[val]
            else:
                last_occurrence[val] = old_index
            current_start = prev_start
        
        dfs(0)
        
        return [max_length, min_nodes if max_length != 0 else 1]