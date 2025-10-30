class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7

        def count_set_bits(n):
            return bin(n).count('1')

        n = int(s, 2)
        dp = [[0] * (k + 1) for _ in range(n)]

        for i in range(1, n):
            dp[i][0] = 1

        for i in range(1, n):
            for j in range(1, k + 1):
                next_val = count_set_bits(i)
                if next_val == 1:
                    dp[i][j] = (dp[i][j] + 1) % MOD
                else:
                    dp[i][j] = (dp[i][j] + dp[next_val][j - 1]) % MOD

        result = 0
        for j in range(k + 1):
            result = (result + dp[n - 1][j]) % MOD

        return result