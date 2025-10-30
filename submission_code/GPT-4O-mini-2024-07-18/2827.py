from typing import List
from math import gcd
from collections import defaultdict

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return True
        
        # Create a graph where each number points to its indices
        graph = defaultdict(list)
        
        # Build the graph based on gcd condition
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(nums[i], nums[j]) > 1:
                    graph[i].append(j)
                    graph[j].append(i)
        
        # Function to perform DFS
        def dfs(node, visited):
            stack = [node]
            while stack:
                current = stack.pop()
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
        
        # Check connectivity from the first index
        visited = set()
        visited.add(0)
        dfs(0, visited)
        
        # If we visited all indices, return True
        return len(visited) == n