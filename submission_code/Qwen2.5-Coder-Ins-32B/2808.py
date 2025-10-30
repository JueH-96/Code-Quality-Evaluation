from typing import List
from functools import lru_cache

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        
        @lru_cache(None)
        def dp(i, remaining):
            if remaining <= 0:
                return 0
            if i == n:
                return float('inf')
            
            # Option 1: Hire the paid painter for wall i
            hire_paid = cost[i] + dp(i + 1, remaining - 1 - time[i])
            
            # Option 2: Use the free painter for wall i
            use_free = dp(i + 1, remaining)
            
            return min(hire_paid, use_free)
        
        return dp(0, n)