from typing import List
from functools import lru_cache

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        dp = [1] * n
        coordinates.sort()
        
        @lru_cache(None)
        def dfs(index):
            if index == k:
                return 1
            max_length = 1
            for next_index in range(index + 1, n):
                if coordinates[next_index][0] > coordinates[index][0] and coordinates[next_index][1] > coordinates[index][1]:
                    max_length = max(max_length, 1 + dfs(next_index))
            return max_length
        
        return max(dfs(i) for i in range(n))