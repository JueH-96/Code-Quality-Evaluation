class Solution:
    def waysToReachStair(self, k: int) -> int:
        MOD = 10**9 + 7
        memo = {}
        
        def dp(stair, jump, last_op):
            # Base case: reached the target
            if stair == k:
                return 1
            
            # Pruning to avoid infinite recursion
            if stair < 0 or jump > 35:  # log2(10^9) ≈ 30, added buffer
                return 0
            
            # Additional pruning for efficiency
            if stair > k + (1 << 35):  # If we've gone too far up
                return 0
                
            # Check memo
            if (stair, jump, last_op) in memo:
                return memo[(stair, jump, last_op)]
            
            ways = 0
            
            # Option 1: Go down (if allowed)
            # last_op=1 means previous operation was up or it's the start
            if last_op == 1 and stair > 0:
                ways = (ways + dp(stair - 1, jump, 0)) % MOD
            
            # Option 2: Go up
            next_stair = stair + (1 << jump)
            ways = (ways + dp(next_stair, jump + 1, 1)) % MOD
            
            memo[(stair, jump, last_op)] = ways
            return ways
        
        # Start at stair 1 with jump=0 and last_op=1 (representing start state)
        return dp(1, 0, 1)