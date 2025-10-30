class Solution:
    def waysToReachStair(self, k: int) -> int:
        MOD = 10**9 + 7
        
        @cache
        def dfs(stair, jump, prev_down):
            if stair == k:
                return 1
            if stair > k:
                return 0
            
            total = 0
            
            # Go up
            total += dfs(stair + (1 << jump), jump + 1, False)
            
            # Go down
            if not prev_down and stair > 0:
                total += dfs(stair - 1, jump, True)
            
            return total % MOD
        
        return dfs(1, 0, False)