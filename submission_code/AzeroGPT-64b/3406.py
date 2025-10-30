class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10 ** 9 + 7
        
        @lru_cache(None)
        def dp(i, zs, lis):
            if i == 0:
                return 1 if (not zs) and lis == 1 else 0
            if lis + i <= limit:
                return (dp(i - 1, zs, lis + 1) + dp(i - 1, zs - 1, 1)) % MOD
            return dp(i - 1, zs, lis + 1)
        
        res = 0
        for i in range(1, one + 1):
            res = (res + dp(zero, zero, i)) % MOD
        
        return res