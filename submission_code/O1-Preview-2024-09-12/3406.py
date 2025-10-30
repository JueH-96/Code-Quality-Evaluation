class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9+7
        
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dp(z, o, l, c):
            if z == 0 and o == 0:
                return 1
            total = 0
            if c == 0:
                # last symbol was 0
                if z > 0:
                    if l < limit:
                        # Can place another 0
                        total += dp(z-1, o, l+1, 0)
                    # Switch to 1
                    if o > 0:
                        total += dp(z, o-1, 1, 1)
                else:
                    # No more zeros, can only switch to ones
                    if o > 0:
                        total += dp(z, o-1, 1, 1)
            else:
                # last symbol was 1
                if o > 0:
                    if l < limit:
                        # Can place another 1
                        total += dp(z, o-1, l+1, 1)
                    # Switch to 0
                    if z > 0:
                        total += dp(z-1, o, 1, 0)
                else:
                    # No more ones, can only switch to zeros
                    if z > 0:
                        total += dp(z-1, o, 1, 0)
            total %= MOD
            return total

        total = 0
        if zero > 0:
            total += dp(zero-1, one, 1, 0)
        if one > 0:
            total += dp(zero, one-1, 1, 1)
        return total % MOD