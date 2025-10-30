class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        max_num = 1
        while (max_num ** x) <= n:
            max_num += 1
        max_num -= 1
        dp = [0] * (n + 1)
        dp[0] = 1
        for num in range(1, max_num + 1):
            power = num ** x
            for i in range(n, power - 1, -1):
                dp[i] = (dp[i] + dp[i - power]) % MOD
        return dp[n]