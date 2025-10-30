from typing import List
from math import gcd
from collections import defaultdict

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        graph = defaultdict(list)
        for i in range(n):
            for j in range(i+1, n):
                if gcd(nums[i], nums[j]) > 1:
                    graph[i].append(j)
                    graph[j].append(i)
        visited = [False]*n
        def dfs(node):
            if visited[node]:
                return
            visited[node] = True
            for nei in graph[node]:
                dfs(nei)
        dfs(0)
        return all(visited)