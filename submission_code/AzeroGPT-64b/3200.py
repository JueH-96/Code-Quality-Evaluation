class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        if n < 4:
            return 0

        # Dynamic programming approach
        dp = [0] * (n + 1)
        dp[0] = 1  # Empty string is considered 1 way
        for i in range(1, n + 1):
            dp[i] = (26 * dp[i - 1]) % MOD
            if i >= 4:
                dp[i] = (dp[i] - (26 ** 4 - 16) * dp[i - 4]) % MOD
            if i >= 5:
                dp[i] = (dp[i] + 4 * 25 * (26 ** 3 - 16) * dp[i - 5]) % MOD
            if i >= 6:
                dp[i] = (dp[i] - 6 * 25 * 24 * (26 ** 2 - 16) * dp[i - 6]) % MOD
            if i >= 7:
                dp[i] = (dp[i] + 4 * 25 * 24 * 23 * (26 - 16) * dp[i - 7]) % MOD

        # Total strings minus strings not containing "leet" substring
        total_strings = (26 ** n) % MOD
        return (total_strings - dp[n]) % MOD