class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        ones = [0] * n
        zeros = [0] * n
        for i in range(n):
            if i > 0:
                ones[i] = ones[i-1]
                zeros[i] = zeros[i-1]
            if s[i] == '1':
                ones[i] += 1
            else:
                zeros[i] += 1
        dp = [0] + [float('inf')] * ones[-1]
        for i in range(n-1, -1, -1):
            dp2 = [0] + [float('inf')] * ones[i]
            for j in range(ones[i]+1):
                dp2[j] = min(dp2[j], dp[j-1]+1 if j > 0 else float('inf'))
                dp2[j] = min(dp2[j], dp[j]+1 if j < ones[i] else float('inf'))
                dp2[j] = min(dp2[j], dp[j]+zeros[i]-j)
            dp = dp2
        return dp[-1]