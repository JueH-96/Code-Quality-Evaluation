from typing import List
from math import gcd

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        graph = [[] for _ in range(n)]
        
        # Build the graph
        for i in range(n):
            for j in range(i+1, n):
                if gcd(nums[i], nums[j]) > 1:
                    graph[i].append(j)
                    graph[j].append(i)
        
        # Perform DFS to check if all nodes are connected
        visited = [False] * n
        
        def dfs(node):
            visited[node] = True
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        dfs(0)
        return all(visited)