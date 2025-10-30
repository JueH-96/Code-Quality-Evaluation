class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        
        # Memoization cache
        memo = {}
        
        def dp(i, remaining):
            # Base cases
            if remaining <= 0:
                return 0
            if i >= n:
                return float('inf')  # Can't paint remaining walls
            
            if (i, remaining) in memo:
                return memo[(i, remaining)]
            
            # Option 1: Use paid painter for wall i
            # This wall is painted by paid painter, and during time[i] units,
            # free painter can paint min(time[i], remaining-1) additional walls
            paint_with_paid = cost[i] + dp(i + 1, remaining - 1 - time[i])
            
            # Option 2: Don't use paid painter for wall i (skip this wall for paid painter)
            skip = dp(i + 1, remaining)
            
            result = min(paint_with_paid, skip)
            memo[(i, remaining)] = result
            return result
        
        return dp(0, n)