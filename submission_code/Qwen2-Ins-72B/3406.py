class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        if limit >= one + zero:
            return 2 * factorial(one + zero - 1) // (factorial(one) * factorial(zero - 1)) % MOD
        dp = [0] * (one + zero + 1)
        dp[0] = 1
        for i in range(1, one + 1):
            for j in range(one + zero, i - 1, -1):
                dp[j] = (dp[j] + dp[j - 1]) % MOD
                if j - limit - 1 >= 0:
                    dp[j] = (dp[j] - dp[j - limit - 1]) % MOD
        return dp[one + zero] * pow(2, zero - 1, MOD) % MOD