class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        @cache
        def dp(z, o, last, last_count):
            if z == 0 and o == 0:
                return 1
            if z < 0 or o < 0:
                return 0
            
            count = 0
            if last == 0:
                if last_count < limit:
                    count += dp(z, o, 0, last_count + 1)
                count += dp(z, o - 1, 1, 1)
            else:
                if last_count < limit:
                    count += dp(z, o, 1, last_count + 1)
                count += dp(z - 1, o, 0, 1)
            
            return count % MOD
        
        return dp(zero, one, -1, 0)