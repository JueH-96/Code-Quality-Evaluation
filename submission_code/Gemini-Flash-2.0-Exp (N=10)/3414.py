class Solution:
    def waysToReachStair(self, k: int) -> int:
        if k == 0:
            return 2
        
        memo = {}
        
        def solve(curr_stair, jump, last_down):
            if curr_stair == k:
                return 1
            
            if (curr_stair, jump, last_down) in memo:
                return memo[(curr_stair, jump, last_down)]
            
            ways = 0
            
            # Option 1: Go down
            if curr_stair > 0 and not last_down:
                ways += solve(curr_stair - 1, jump, True)
            
            # Option 2: Go up
            ways += solve(curr_stair + (1 << jump), jump + 1, False)
            
            memo[(curr_stair, jump, last_down)] = ways
            return ways
        
        return solve(1, 0, False)