class Solution:
    def waysToReachStair(self, k: int) -> int:
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(stair, jump, last_was_down):
            ways = 0
            
            # Count if we're at stair k
            if stair == k:
                ways += 1
            
            # Pruning: if we're too far above k, we might not be able to reach k again
            # The maximum we can go down is limited by the number of up operations
            # After j up operations, we can go down at most j times
            # So if stair - j > k, we can't reach k anymore
            if stair - jump > k:
                return ways
            
            # Option 1: Go down (if allowed)
            if not last_was_down and stair > 0:
                ways += dp(stair - 1, jump, True)
            
            # Option 2: Go up
            ways += dp(stair + (1 << jump), jump + 1, False)
            
            return ways
        
        return dp(1, 0, False)