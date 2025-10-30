class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        mod = 10**9 + 7
        # dp[i] will hold the value at position i after current number of seconds
        dp = [1] * n
        # Each second, we replace dp[i] with the prefix sum up to i
        for _ in range(k):
            for i in range(1, n):
                dp[i] = (dp[i] + dp[i-1]) % mod
        return dp[-1]