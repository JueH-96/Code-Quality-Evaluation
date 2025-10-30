class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        dp = [[n]*k for _ in range(n)]
        palin = [[0]*n for _ in range(n)]
        for i in range(n):
            dp[i][0] = k-1 if s[i] != s[0] else 0
            for j in range(1, min(i+1, k)):
                dp[i][j] = dp[i-1][j-1] + (s[i] != s[i-j])
        for l in range(2, n+1):
            for i in range(n-l+1):
                j = i+l-1
                if s[i] == s[j]:
                    dp[j][l%k] = min(dp[j][l%k], dp[j-1][(l-1)%k])
                else:
                    dp[j][l%k] = min(dp[j][l%k], dp[j-1][(l-1)%k]+1)
                for m in range(i+1, j):
                    if s[i] == s[m]:
                        dp[j][l%k] = min(dp[j][l%k], dp[m-1][(l-1)%k]+(j-m))
                    else:
                        dp[j][l%k] = min(dp[j][l%k], dp[m-1][(l-1)%k]+(j-m)+1)
        return dp[-1][k%k]