from functools import lru_cache

class Solution:
    def minCost(self, nums: List[int]) -> int:
        @lru_cache(None)
        def dp(remaining):
            n = len(remaining)
            
            # Base cases
            if n == 0:
                return 0
            if n == 1:
                return remaining[0]
            if n == 2:
                return max(remaining)
            
            # We have at least 3 elements
            min_cost = float('inf')
            
            # Option 1: Remove elements at index 0 and 1
            cost = max(remaining[0], remaining[1]) + dp(remaining[2:])
            min_cost = min(min_cost, cost)
            
            # Option 2: Remove elements at index 0 and 2
            new_remaining = (remaining[1],) + remaining[3:]
            cost = max(remaining[0], remaining[2]) + dp(new_remaining)
            min_cost = min(min_cost, cost)
            
            # Option 3: Remove elements at index 1 and 2
            new_remaining = (remaining[0],) + remaining[3:]
            cost = max(remaining[1], remaining[2]) + dp(new_remaining)
            min_cost = min(min_cost, cost)
            
            return min_cost
        
        return dp(tuple(nums))