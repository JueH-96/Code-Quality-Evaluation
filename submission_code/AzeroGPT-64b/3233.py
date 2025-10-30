class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        dp = [0] * n
        skip = 0
        for i in range(n):
            if i - skip >= k:
                dp[i] = 1 + dp[i - k]
            else:
                dp[i] = 1
            for j in range(i):
                if len(set(s[j:i+1])) <= k:
                    skip = max(skip, i - j)
                    dp[i] = max(dp[i], dp[j] + (i != j))
        best = 0
        for i in range(1, n):
            if len(set(s[:i] + s[i] + s[i+1:])) <= k:
                best = max(best, 1 + dp[i-1])
        return best