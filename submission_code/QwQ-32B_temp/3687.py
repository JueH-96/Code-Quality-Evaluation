from typing import List
from collections import deque

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        max_length = 0
        min_nodes = float('inf')
        
        # Determine parent pointers using BFS to build the tree structure
        parent = [-1] * n
        q = deque()
        q.append(0)
        parent[0] = -1  # root has no parent
        
        while q:
            u = q.popleft()
            for v, w in adj[u]:
                if parent[v] == -1 and v != parent[u]:
                    parent[v] = u
                    q.append(v)
        
        # DFS function to explore paths starting from 'start'
        def dfs(u, path_set, current_length, current_nodes):
            nonlocal max_length, min_nodes
            if current_length > max_length:
                max_length = current_length
                min_nodes = current_nodes
            elif current_length == max_length:
                if current_nodes < min_nodes:
                    min_nodes = current_nodes
            
            for v, w in adj[u]:
                if v == parent[u]:
                    continue  # avoid going back to parent
                val = nums[v]
                if val not in path_set:
                    new_set = path_set.copy()
                    new_set.add(val)
                    dfs(v, new_set, current_length + w, current_nodes + 1)
        
        # Start DFS from every node as the starting point
        for start in range(n):
            initial_set = {nums[start]}
            dfs(start, initial_set, 0, 1)
        
        return [max_length, min_nodes]