class Solution:
    def waysToReachStair(self, k: int) -> int:
        MOD = 10**9 + 7
        
        # Cache for memoization
        # State: (current_stair, prev_was_down, jump)
        @cache
        def dp(curr: int, prev_down: bool, jump: int) -> int:
            # Base cases
            if curr < 0:
                return 0
                
            # Found a valid path
            if curr == k:
                return 1
                
            # If we go too far, no valid path
            if curr > k + 100:  # Arbitrary large enough bound
                return 0
                
            ways = 0
            
            # Try going down if not at stair 0 and previous move wasn't down
            if curr > 0 and not prev_down:
                ways = (ways + dp(curr - 1, True, jump)) % MOD
                
            # Try going up using current jump power
            ways = (ways + dp(curr + (1 << jump), False, jump + 1)) % MOD
            
            return ways
            
        return dp(1, False, 0)  # Start at stair 1 with jump = 0