class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)

        # dp[i][j] will store the number of ways to form a valid sequence of length i ending with creature j
        dp = [[0, 0, 0] for _ in range(n + 1)]

        # Base case: one way to have an empty sequence
        dp[0][0] = dp[0][1] = dp[0][2] = 1

        for i in range(1, n + 1):
            if s[i - 1] == 'F':
                dp[i][0] = (dp[i - 1][1] + dp[i - 1][2]) % MOD
                dp[i][1] = dp[i - 1][0]
                dp[i][2] = dp[i - 1][1]
            elif s[i - 1] == 'W':
                dp[i][0] = dp[i - 1][2]
                dp[i][1] = (dp[i - 1][0] + dp[i - 1][2]) % MOD
                dp[i][2] = dp[i - 1][0]
            elif s[i - 1] == 'E':
                dp[i][0] = dp[i - 1][1]
                dp[i][1] = dp[i - 1][2]
                dp[i][2] = (dp[i - 1][0] + dp[i - 1][1]) % MOD

        # Summing up all valid sequences of length n
        total_sequences = (dp[n][0] + dp[n][1] + dp[n][2]) % MOD

        return total_sequences