class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        if n == 0:
            return 0
        
        # Initialize DP tables for each child
        # dp1: child starting at (0, 0)
        # dp2: child starting at (0, n-1)
        # dp3: child starting at (n-1, 0)
        
        dp1 = [[0] * n for _ in range(n)]
        dp2 = [[0] * n for _ in range(n)]
        dp3 = [[0] * n for _ in range(n)]
        
        # Initialize starting positions
        dp1[0][0] = fruits[0][0]
        dp2[0][n-1] = fruits[0][n-1]
        dp3[n-1][0] = fruits[n-1][0]
        
        # Fill dp1
        for i in range(n):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                if i > 0 and j > 0:
                    dp1[i][j] = max(dp1[i][j], dp1[i-1][j-1] + fruits[i][j])
                if i > 0:
                    dp1[i][j] = max(dp1[i][j], dp1[i-1][j] + fruits[i][j])
                if j > 0:
                    dp1[i][j] = max(dp1[i][j], dp1[i][j-1] + fruits[i][j])
        
        # Fill dp2
        for i in range(n):
            for j in range(n-1, -1, -1):
                if i == 0 and j == n-1:
                    continue
                if i > 0 and j < n-1:
                    dp2[i][j] = max(dp2[i][j], dp2[i-1][j+1] + fruits[i][j])
                if i > 0:
                    dp2[i][j] = max(dp2[i][j], dp2[i-1][j] + fruits[i][j])
                if j < n-1:
                    dp2[i][j] = max(dp2[i][j], dp2[i][j+1] + fruits[i][j])
        
        # Fill dp3
        for i in range(n-1, -1, -1):
            for j in range(n):
                if i == n-1 and j == 0:
                    continue
                if i < n-1 and j > 0:
                    dp3[i][j] = max(dp3[i][j], dp3[i+1][j-1] + fruits[i][j])
                if j > 0:
                    dp3[i][j] = max(dp3[i][j], dp3[i][j-1] + fruits[i][j])
                if i < n-1:
                    dp3[i][j] = max(dp3[i][j], dp3[i+1][j] + fruits[i][j])
        
        # Now, calculate the maximum total fruits collected
        max_fruits = 0
        for i in range(n):
            for j in range(n):
                total = dp1[i][j] + dp2[i][j] + dp3[i][j]
                # If all three children are at the same cell, subtract the fruits twice
                # because they are counted three times but only collected once
                if dp1[i][j] > 0 and dp2[i][j] > 0 and dp3[i][j] > 0:
                    total -= 2 * fruits[i][j]
                max_fruits = max(max_fruits, total)
        
        return max_fruits