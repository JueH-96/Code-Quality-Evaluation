class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        diff = []
        for i in range(n):
            if s1[i] != s2[i]:
                diff.append(i)
        m = len(diff)
        if m % 2 != 0:
            return -1
        if m == 0:
            return 0
        # Initialize DP array
        dp = [float('inf')] * (m + 1)
        dp[0] = 0
        for i in range(1, m + 1):
            if i >= 2:
                # Use the second operation for the last two differences
                cost = diff[i-1] - diff[i-2]
                dp[i] = min(dp[i], dp[i-2] + cost)
            if i >= 1:
                # Use the first operation for the last difference
                dp[i] = min(dp[i], dp[i-1] + x)
        return dp[m] if dp[m] != float('inf') else -1