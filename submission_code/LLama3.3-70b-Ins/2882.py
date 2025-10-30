class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        memo = {}

        def dp(n, x, max_num):
            if (n, max_num) in memo:
                return memo[(n, max_num)]
            if n == 0:
                return 1
            if n < 0 or max_num <= 0:
                return 0
            res = dp(n, x, max_num - 1)
            if max_num ** x <= n:
                res += dp(n - max_num ** x, x, max_num - 1)
            memo[(n, max_num)] = res % MOD
            return memo[(n, max_num)]

        return dp(n, x, n)