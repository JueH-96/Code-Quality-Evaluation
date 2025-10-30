from functools import lru_cache

class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        @lru_cache(maxsize=None)
        def dp(a, b, c, s):
            res = 0
            if s == -1:
                if a > 0:
                    res = max(res, 2 + dp(a-1, b, c, 0))
                if b > 0:
                    res = max(res, 2 + dp(a, b-1, c, 1))
                if c > 0:
                    res = max(res, 2 + dp(a, b, c-1, 2))
            elif s == 0:
                if b > 0:
                    res = max(res, 2 + dp(a, b-1, c, 1))
            elif s == 1:
                if a > 0:
                    res = max(res, 2 + dp(a-1, b, c, 0))
                if c > 0:
                    res = max(res, 2 + dp(a, b, c-1, 2))
            elif s == 2:
                if a > 0:
                    res = max(res, 2 + dp(a-1, b, c, 0))
                if c > 0:
                    res = max(res, 2 + dp(a, b, c-1, 2))
            return res
        return dp(x, y, z, -1)