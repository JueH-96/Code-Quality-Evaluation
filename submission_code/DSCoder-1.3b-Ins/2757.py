class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        num1, num2 = int(num1), int(num2)
        dp = [[0] * (max_sum + 1) for _ in range(10)]
        dp[0][0] = 1
        for i in range(10):
            for j in range(max_sum + 1):
                for k in range(1, j // min_sum + 1):
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - k * min_sum]) % MOD
        res = 0
        for i in range(num1, num2 + 1):
            temp = [0] * (max_sum + 1)
            temp[0] = 1
            for j in range(1, len(str(i))):
                for k in range(len(str(i)) - j):
                    temp[k + 1] = (temp[k + 1] + temp[k] * int(str(i)[j])) % MOD
            for j in range(min_sum, max_sum + 1):
                res = (res + temp[len(str(i)) - len(str(num1))] * dp[len(str(i)) - len(str(num1))][j]) % MOD
        return res