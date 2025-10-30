class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        @lru_cache(None)
        def dp(z, o, last):
            if z == 0 and o == 0:
                return 1
            if z < 0 or o < 0:
                return 0
            
            ans = 0
            if z > 0:
                ans += dp(z - 1, o, 0) if last == 1 or last == -1 else 0
            if o > 0:
                ans += dp(z, o - 1, 1) if last == 0 or last == -1 else 0
            
            if last == 0 and z > limit:
                ans -= dp(z - limit - 1, o, 0)
            if last == 1 and o > limit:
                ans -= dp(z, o - limit - 1, 1)
            
            return ans % MOD
        
        return (dp(zero, one, 0) + dp(zero, one, 1)) % MOD