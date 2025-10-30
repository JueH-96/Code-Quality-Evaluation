class Solution:
    def waysToReachStair(self, k: int) -> int:
        from functools import lru_cache
        
        @lru_cache(None)
        def dfs(pos, jump, last_was_down):
            # Base cases
            if pos < 0:
                return 0
            
            # If jump is too large, we've gone too far
            # With jump value j, we can go up by at least 2^j
            # If current pos + 2^j - j > k + 40, we've likely gone too far
            if jump > 40:  # 2^40 is extremely large
                return 0
            
            ways = 0
            
            # If current position is k, we found one way
            if pos == k:
                ways = 1
            
            # Try going down (if allowed)
            if not last_was_down and pos > 0:
                ways += dfs(pos - 1, jump, True)
            
            # Try going up
            ways += dfs(pos + (1 << jump), jump + 1, False)
            
            return ways
        
        # Start from position 1, jump = 0, last operation was not down
        return dfs(1, 0, False)