class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(a, b, c, state):
            res = 0
            if a > 0:
                if state == 0 or state == 3 or state == 4:
                    res = max(res, 2 + dp(a-1, b, c, 2))
            if b > 0:
                if state == 0 or state == 1 or state == 2:
                    res = max(res, 2 + dp(a, b-1, c, 4))
            if c > 0:
                if state != 2:
                    res = max(res, 2 + dp(a, b, c-1, 3))
            return res
        
        return dp(x, y, z, 0)