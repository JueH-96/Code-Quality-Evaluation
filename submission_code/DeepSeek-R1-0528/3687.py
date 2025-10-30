import sys
from collections import deque

sys.setrecursionlimit(100000)

class Solution:
    def longestSpecialPath(self, edges: list, nums: list) -> list:
        n = len(nums)
        graph = [[] for _ in range(n)]
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        depth_arr = [-1] * n
        cum_arr = [0] * n
        q = deque([0])
        depth_arr[0] = 0
        while q:
            u = q.popleft()
            for v, w in graph[u]:
                if depth_arr[v] == -1:
                    depth_arr[v] = depth_arr[u] + 1
                    cum_arr[v] = cum_arr[u] + w
                    q.append(v)
        
        path = [0] * n
        last_occurrence = {}
        best_length = 0
        min_nodes = float('inf')
        
        def dfs(u, parent, start_index):
            nonlocal best_length, min_nodes, path, last_occurrence
            d = depth_arr[u]
            path[d] = u
            
            old_occurrence = last_occurrence.get(nums[u], -1)
            new_start_index = start_index
            if old_occurrence != -1:
                new_start_index = max(new_start_index, old_occurrence + 1)
            last_occurrence[nums[u]] = d
            
            start_node = path[new_start_index]
            current_length = cum_arr[u] - cum_arr[start_node]
            current_nodes = d - new_start_index + 1
            
            if current_length > best_length:
                best_length = current_length
                min_nodes = current_nodes
            elif current_length == best_length and current_nodes < min_nodes:
                min_nodes = current_nodes
                
            for v, w in graph[u]:
                if v == parent:
                    continue
                dfs(v, u, new_start_index)
                
            if old_occurrence == -1:
                if nums[u] in last_occurrence:
                    del last_occurrence[nums[u]]
            else:
                last_occurrence[nums[u]] = old_occurrence
        
        dfs(0, -1, 0)
        
        return [best_length, min_nodes]