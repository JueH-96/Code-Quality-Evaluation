class Solution:
    def waysToReachStair(self, k: int) -> int:
        from functools import lru_cache
        
        @lru_cache(None)
        def dfs(pos, jump, can_go_down):
            # If we're at position k, this counts as one way
            if pos == k:
                count = 1
            else:
                count = 0
            
            # Pruning: stop if jump value is too large
            # Since k <= 10^9 < 2^31, we need at most 31 jumps
            if jump > 31:
                return count
            
            # Try going down (if allowed)
            if can_go_down and pos > 0:
                count += dfs(pos - 1, jump, False)
            
            # Try going up
            count += dfs(pos + (1 << jump), jump + 1, True)
            
            return count
        
        return dfs(1, 0, True)