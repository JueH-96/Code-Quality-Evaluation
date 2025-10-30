class Solution:
    def waysToReachStair(self, k: int) -> int:
        memo = {}
        
        def dp(i, j, can_go_down):
            if (i, j, can_go_down) in memo:
                return memo[(i, j, can_go_down)]
            
            # Pruning: if we can't reach k from this state
            min_reachable = i - 1 if can_go_down and i > 0 else i
            if min_reachable > k:
                memo[(i, j, can_go_down)] = 0
                return 0
            
            ways = 0
            
            # Count this position if it's the target
            if i == k:
                ways += 1
            
            # Option 1: go down (if allowed)
            if can_go_down and i > 0:
                ways += dp(i - 1, j, False)
            
            # Option 2: go up (if it's worth exploring)
            next_stair = i + (1 << j)
            if next_stair <= k + 1:
                ways += dp(next_stair, j + 1, True)
            
            memo[(i, j, can_go_down)] = ways
            return ways
        
        return dp(1, 0, True)