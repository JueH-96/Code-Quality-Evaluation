class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        cnt = [[0]*26 for _ in range(n+1)]
        for i in range(n):
            cnt[i+1] = cnt[i][:]
            cnt[i+1][ord(s[i])-97] += 1
        dp = [0]*(n+1)
        for i in range(n-1, -1, -1):
            dp[i] = max(dp[i+1], dp[cnt[i+1][ord(s[i])-97]+1]+1 if cnt[i+1][ord(s[i])-97]-cnt[i][ord(s[i])-97]==1 else 0)
        return dp[0] if dp[0]>=3 else -1