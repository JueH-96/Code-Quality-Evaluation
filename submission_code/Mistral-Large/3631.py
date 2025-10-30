class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n = int(s, 2)

        # DP table to store the number of k-reducible integers
        dp = [[0] * (k + 1) for _ in range(n + 1)]

        # Base case: 1 is always k-reducible
        for j in range(k + 1):
            dp[1][j] = 1

        # Fill the DP table
        for i in range(2, n + 1):
            count_set_bits = bin(i).count('1')
            if count_set_bits <= n:
                for j in range(1, k + 1):
                    dp[count_set_bits][j] = (dp[count_set_bits][j] + dp[i][j - 1]) % MOD

        # Sum up the results
        result = 0
        for j in range(k + 1):
            result = (result + dp[1][j]) % MOD

        return result