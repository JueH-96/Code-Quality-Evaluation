class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        
        # dp[i][j] represents the time when wizard i finishes working on potion j
        dp = [[0] * m for _ in range(n)]
        
        # Base case: wizard 0 working on potion 0
        dp[0][0] = skill[0] * mana[0]
        
        # Fill first row: wizard 0 working on all potions
        for j in range(1, m):
            dp[0][j] = dp[0][j-1] + skill[0] * mana[j]
        
        # Fill first column: all wizards working on potion 0
        for i in range(1, n):
            dp[i][0] = dp[i-1][0] + skill[i] * mana[0]
        
        # Fill the rest of the table
        for i in range(1, n):
            for j in range(1, m):
                # Wizard i can start working on potion j when:
                # 1. Previous wizard (i-1) finished with potion j
                # 2. Wizard i finished with previous potion (j-1)
                start_time = max(dp[i-1][j], dp[i][j-1])
                dp[i][j] = start_time + skill[i] * mana[j]
        
        # The answer is when the last wizard finishes the last potion
        return dp[n-1][m-1]