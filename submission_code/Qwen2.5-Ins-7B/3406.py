class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        @lru_cache(None)
        def dp(z, o, l):
            if l == 0:
                return z == 0 and o == 1
            if z == 0 and o == 1:
                return 1
            if z == 0 or o == 1:
                return 0
            return dp(z-1, o, l-1) + dp(z, o-1, l-1)
        
        return dp(zero, one, limit) % MOD