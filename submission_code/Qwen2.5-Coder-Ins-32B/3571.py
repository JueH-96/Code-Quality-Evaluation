from typing import List
from functools import lru_cache

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        @lru_cache(None)
        def dfs(x, y):
            max_length = 1
            for nx, ny in coordinates:
                if nx > x and ny > y:
                    max_length = max(max_length, 1 + dfs(nx, ny))
            return max_length
        
        # Sort coordinates to ensure we process in a consistent order
        coordinates.sort()
        start_x, start_y = coordinates[k]
        return dfs(start_x, start_y)