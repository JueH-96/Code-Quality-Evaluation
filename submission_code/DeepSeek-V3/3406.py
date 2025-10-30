class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dp(z, o, last_char, last_count):
            if z == 0 and o == 0:
                return 1
            total = 0
            if z > 0:
                if last_char == 0:
                    if last_count < limit:
                        total += dp(z-1, o, 0, last_count + 1)
                else:
                    total += dp(z-1, o, 0, 1)
                total %= MOD
            if o > 0:
                if last_char == 1:
                    if last_count < limit:
                        total += dp(z, o-1, 1, last_count + 1)
                else:
                    total += dp(z, o-1, 1, 1)
                total %= MOD
            return total

        return dp(zero, one, -1, 0) % MOD