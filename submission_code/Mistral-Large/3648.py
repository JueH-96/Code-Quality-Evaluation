from typing import List

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)

        # Initialize dp arrays for each child
        dp1 = [[0] * n for _ in range(n)]
        dp2 = [[0] * n for _ in range(n)]
        dp3 = [[0] * n for _ in range(n)]

        # Initialize the starting points
        dp1[0][0] = fruits[0][0]
        dp2[0][n-1] = fruits[0][n-1]
        dp3[n-1][0] = fruits[n-1][0]

        # Fill the dp arrays
        for i in range(n):
            for j in range(n):
                if i > 0 and j > 0:
                    dp1[i][j] = max(dp1[i-1][j-1], dp1[i-1][j], dp1[i][j-1]) + fruits[i][j]
                elif i > 0:
                    dp1[i][j] = dp1[i-1][j] + fruits[i][j]
                elif j > 0:
                    dp1[i][j] = dp1[i][j-1] + fruits[i][j]

                if i > 0 and j < n-1:
                    dp2[i][j] = max(dp2[i-1][j+1], dp2[i-1][j], dp2[i-1][j-1]) + fruits[i][j]
                elif i > 0:
                    dp2[i][j] = dp2[i-1][j] + fruits[i][j]
                elif j < n-1:
                    dp2[i][j] = dp2[i-1][j+1] + fruits[i][j]

                if i < n-1 and j > 0:
                    dp3[i][j] = max(dp3[i+1][j-1], dp3[i][j-1], dp3[i+1][j+1]) + fruits[i][j]
                elif i < n-1:
                    dp3[i][j] = dp3[i+1][j-1] + fruits[i][j]
                elif j > 0:
                    dp3[i][j] = dp3[i][j-1] + fruits[i][j]

        # Calculate the maximum fruits collected
        max_fruits = max(dp1[n-1][n-1], dp2[n-1][n-1], dp3[n-1][n-1])

        return max_fruits