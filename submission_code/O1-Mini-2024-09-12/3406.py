class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        from functools import lru_cache

        @lru_cache(None)
        def dp(z, o, last, cnt):
            if z == 0 and o == 0:
                return 1
            res = 0
            if z > 0:
                if last == 0:
                    if cnt < limit:
                        res += dp(z-1, o, 0, cnt+1)
                else:
                    res += dp(z-1, o, 0, 1)
            if o > 0:
                if last == 1:
                    if cnt < limit:
                        res += dp(z, o-1, 1, cnt+1)
                else:
                    res += dp(z, o-1, 1, 1)
            return res % MOD

        return dp(zero, one, -1, 0)