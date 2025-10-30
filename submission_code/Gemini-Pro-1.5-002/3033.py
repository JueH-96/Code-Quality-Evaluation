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
        
        if x >= n:
            return m // 2
        
        dp = [[float('inf')] * (m + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        
        for i in range(m):
            for j in range(i + 1):
                cost = 0
                if j > 0:
                    cost = dp[i][j-1] + x
                else:
                    cost = dp[i][j] + x
                dp[i+1][j] = min(dp[i+1][j], cost)
                
                if i + 1 < m:
                    cost = 0
                    if diff[i+1] == diff[i] + 1:
                        if j > 0:
                            cost = dp[i][j-1] + 1
                        else:
                            cost = dp[i][j] + 1
                    else:
                        cost = dp[i][j] + x
                    dp[i+1][j+1] = min(dp[i+1][j+1], cost)
                
        if dp[m][m//2] == float('inf'):
            return -1
        return dp[m][m//2]