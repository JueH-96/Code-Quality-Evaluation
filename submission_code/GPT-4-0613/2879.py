class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        dp = [[0]*(k+1) for _ in range(n+1)]
        cost = [[0]*n for _ in range(n)]
        
        for length in range(2, n+1):
            for start in range(n-length+1):
                end = start + length - 1
                cost[start][end] = cost[start+1][end-1] + (s[start] != s[end])
        
        for i in range(1, n+1):
            dp[i][1] = cost[0][i-1]
            for j in range(2, min(i, k)+1):
                dp[i][j] = min(dp[k][j-1] + cost[k][i-1] for k in range(j-1, i))
        
        return dp[n][k]