class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        mod = 10**9 + 7
        dp = [0]*(n+1)
        dp[0] = 1
        for i in range(1, n+1):
            if i in sick:
                dp[i] = dp[i-1]
            else:
                dp[i] = (dp[i-1] + dp[i-2]) % mod
        return dp[n]