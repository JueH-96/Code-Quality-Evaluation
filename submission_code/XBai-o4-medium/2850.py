class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dp(a, b, c, last):
            max_len = 0
            if last == 0:
                # Can add any
                if a >= 1:
                    max_len = max(max_len, 2 + dp(a-1, b, c, 1))
                if b >= 1:
                    max_len = max(max_len, 2 + dp(a, b-1, c, 2))
                if c >= 1:
                    max_len = max(max_len, 2 + dp(a, b, c-1, 3))
            elif last == 1:
                # Can add BB
                if b >= 1:
                    max_len = max(max_len, 2 + dp(a, b-1, c, 2))
            elif last == 2:
                # Can add AA or AB
                if a >= 1:
                    max_len = max(max_len, 2 + dp(a-1, b, c, 1))
                if c >= 1:
                    max_len = max(max_len, 2 + dp(a, b, c-1, 3))
            elif last == 3:
                # Can add AA or AB
                if a >= 1:
                    max_len = max(max_len, 2 + dp(a-1, b, c, 1))
                if c >= 1:
                    max_len = max(max_len, 2 + dp(a, b, c-1, 3))
            return max_len

        return dp(x, y, z, 0)