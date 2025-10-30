class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        
        dp = [[0] * m for _ in range(n)]
        
        dp[0][0] = skill[0] * mana[0]
        
        for i in range(1, n):
            dp[i][0] = dp[i-1][0] + skill[i] * mana[0]
            
        for j in range(1, m):
            dp[0][j] = dp[0][j-1] + skill[0] * mana[j]
            
        for i in range(1, n):
            for j in range(1, m):
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + skill[i] * mana[j]
                
        return dp[n-1][m-1]