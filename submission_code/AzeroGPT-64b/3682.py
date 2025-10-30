class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        mod = 10**9 + 7
        mod_m = m % mod
        
        @lru_cache(None)
        def dp(idx, repeat):
            if idx == n:
                return 1 if repeat == k else 0
            ways = 0
            if repeat:
                ways += dp(idx + 1, repeat - 1) * mod_m
                ways %= mod
            ways += dp(idx + 1, min(k, m - 1)) * ((m - 1) % mod)
            ways %= mod
            return ways
        return dp(0, k)