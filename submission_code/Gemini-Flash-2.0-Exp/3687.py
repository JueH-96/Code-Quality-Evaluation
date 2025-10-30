class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)
        graph = defaultdict(list)
        for u, v, length in edges:
            graph[u].append((v, length))
            graph[v].append((u, length))

        max_len = 0
        min_nodes = float('inf')

        def dfs(node, parent, current_path, current_length, current_nodes):
            nonlocal max_len, min_nodes

            if nums[node] in current_path:
                return

            current_path.add(nums[node])
            
            is_leaf = True
            for neighbor, weight in graph[node]:
                if neighbor != parent:
                    is_leaf = False
                    dfs(neighbor, node, current_path.copy(), current_length + weight, current_nodes + 1)

            if is_leaf:
                if current_length > max_len:
                    max_len = current_length
                    min_nodes = current_nodes
                elif current_length == max_len:
                    min_nodes = min(min_nodes, current_nodes)
            
            if current_length > max_len:
                max_len = current_length
                min_nodes = current_nodes
            elif current_length == max_len:
                min_nodes = min(min_nodes, current_nodes)

        for i in range(n):
            dfs(i, -1, set(), 0, 1)

        if max_len == 0 and min_nodes == float('inf'):
            return [0, 1]
        
        return [max_len, min_nodes]
from collections import defaultdict
from typing import List