class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        dp = [[[0] * n for _ in range(n)] for _ in range(n)]

        # Iterate over the possible steps
        for step in range(n):
            for i in range(max(0, step - (n - 1)), min(n, step + 1)):
                j1 = step - i
                if j1 < 0 or j1 >= n:
                    continue
                for j2 in range(max(0, step - (n - 1)), min(n, step + 1)):
                    j3 = step - j2
                    if j3 < 0 or j3 >= n:
                        continue
                    if i + j1 != j2 + j3:  # Ensure children are not in the same room
                        fruits_collected = (
                            fruits[i][j1] + fruits[j2][j2] + fruits[j3][j3]
                        )
                        dp[i][j1][j2] = max(dp[i][j1][j2], fruits_collected)

        # Collect the maximum fruits collected at the bottom-right corner
        max_fruits = 0
        for i in range(n):
            for j in range(n):
                max_fruits = max(max_fruits, dp[i][j][n - 1])

        return max_fruits