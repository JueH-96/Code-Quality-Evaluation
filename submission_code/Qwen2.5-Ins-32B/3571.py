from typing import List

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        def dfs(i):
            if memo[i] != -1:
                return memo[i]
            memo[i] = 1
            for j in range(n):
                if coordinates[i][0] < coordinates[j][0] and coordinates[i][1] < coordinates[j][1]:
                    memo[i] = max(memo[i], 1 + dfs(j))
            return memo[i]
        
        coordinates.sort()
        n = len(coordinates)
        memo = [-1] * n
        return dfs(k)