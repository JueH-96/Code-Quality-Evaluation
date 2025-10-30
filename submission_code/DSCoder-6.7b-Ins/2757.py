class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        dp = [[0]*12 for _ in range(len(num2)+1)]
        dp[0][0] = 1
        for i in range(len(num2)):
            for j in range(10):
                for k in range(j+1):
                    dp[i+1][j] = (dp[i+1][j] + dp[i][j-k]) % MOD
        sum1, sum2 = 0, 0
        for i in range(len(num1)-1, -1, -1):
            sum1 = (sum1*10 + int(num1[i])) % MOD
            sum2 = (sum2*10 + int(num2[i])) % MOD
        sum1 = (sum1+1) % MOD
        sum2 = (sum2+1) % MOD
        res = 0
        for i in range(1, 11):
            res = (res + (sum2 - sum1)*dp[len(num2)][i]) % MOD
        res = (res - (sum2 - sum1)*dp[len(num2)][0]) % MOD
        res = (res + MOD) % MOD
        return res