from typing import List
from math import gcd
from collections import deque

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        graph = [[] for _ in range(n)]
        
        # Build the graph
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(nums[i], nums[j]) > 1:
                    graph[i].append(j)
                    graph[j].append(i)
        
        # Check if all pairs are connected
        for i in range(n):
            visited = set()
            queue = deque([i])
            visited.add(i)
            
            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # If not all nodes are visited, return False
            if len(visited) != n:
                return False
        
        return True