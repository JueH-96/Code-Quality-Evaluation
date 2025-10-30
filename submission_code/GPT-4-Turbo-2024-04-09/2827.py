from math import gcd
from collections import defaultdict, deque
from typing import List

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return True
        
        # Graph adjacency list
        graph = defaultdict(list)
        
        # Connect nodes with gcd > 1
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(nums[i], nums[j]) > 1:
                    graph[i].append(j)
                    graph[j].append(i)
        
        # BFS to check if all nodes are connected
        visited = [False] * n
        queue = deque([0])
        visited[0] = True
        count = 1
        
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    count += 1
        
        # If all nodes are visited, then it's possible to traverse between all pairs
        return count == n