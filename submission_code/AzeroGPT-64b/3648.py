class Solution:
    def maxCollectedFruits(self, fruits):
        n = len(fruits)
        dp = [[[-1 for _ in range(n)] for _ in range(n)] for _ in range(n)]
        
        def dfs(i1, j1, i2, j2, i3, j3):
            if i1 >= n or i2 >= n or i3 >= n or j1 >= n or j2 >= n or j3 >= n or i1 > i2 or i1 > i3 or j1 > j2 or j1 > j3:
                return float('-inf')
            
            if dp[i1][i2][i3] != -1:
                return dp[i1][i2][i3]
            
            if (i1 == n-1 and j1 == n-1) and (i2 == n-1 and j2 == n-1) and (i3 == n-1 and j3 == n-1):
                return fruits[i1][j1]
            
            ans = fruits[i1][j1] if (i1, j1) not in [(i2, j2), (i3, j3)] else 0
            ans += fruits[i2][j2] if (i2, j2) not in [(i1, j1), (i3, j3)] else 0
            ans += fruits[i3][j3] if (i3, j3) not in [(i1, j1), (i2, j2)] else 0
            
            candidates = [dfs(i1+1, j1+1, i2+1, j2, i3+1, j3), 
                          dfs(i1+1, j1,   i2+1, j2, i3+1, j3), 
                          dfs(i1,   j1+1, i2+1, j2, i3+1, j3),
                          dfs(i1+1, j1+1, i2+1, j2,   i3, j3+1), 
                          dfs(i1+1, j1,   i2+1, j2,   i3, j3+1)]
            candidates += [dfs(i1+1, j1+1,   i2, j2+1, i3+1, j3), 
                           dfs(i1+1, j1,     i2, j2+1, i3+1, j3), 
                           dfs(i1,   j1+1,   i2, j2+1, i3+1, j3),
                           dfs(i1+1, j1+1,   i2, j2+1,   i3, j3+1), 
                           dfs(i1+1, j1,     i2, j2+1,   i3, j3+1)]
            candidates += [dfs(i1+1, j1+1,   i2, j2+1,   i3, j3), 
                           dfs(i1+1, j1,     i2, j2+1,   i3, j3),
                           dfs(i1,   j1+1,   i2, j2+1,   i3, j3)]
            
            dp[i1][i2][i3] = max(candidates) + ans
            return dp[i1][i2][i3]
        
        return dfs(0, 0, 0, n-1, n-1, 0)