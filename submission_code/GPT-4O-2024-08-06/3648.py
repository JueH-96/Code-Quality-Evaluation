class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        
        # dp[i][j][k] will store the maximum fruits collected when:
        # - First child is at (i, j)
        # - Second child is at (i, k)
        # - Third child is at (n-1, i + j - k)
        dp = [[[0] * n for _ in range(n)] for _ in range(n)]
        
        # Initialize the starting positions
        dp[0][0][n-1] = fruits[0][0] + fruits[0][n-1] + fruits[n-1][0]
        
        # Iterate over the possible number of moves
        for move in range(1, n):
            for i in range(move + 1):
                for j in range(move + 1):
                    if i < n and j < n and move - i < n and move - j < n:
                        # Calculate the third child's position
                        k = move - i
                        l = move - j
                        if k < n and l < n:
                            # Calculate the maximum fruits collected for this configuration
                            current_fruits = fruits[i][j] + fruits[k][l]
                            if i == k and j == l:
                                current_fruits -= fruits[i][j]  # Avoid double counting if they are in the same cell
                            
                            # Update dp[i][j][k] considering all possible previous moves
                            max_prev = 0
                            for di, dj, dk in [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]:
                                ni, nj, nk = i - di, j - dj, k - dk
                                if 0 <= ni < n and 0 <= nj < n and 0 <= nk < n:
                                    max_prev = max(max_prev, dp[ni][nj][nk])
                            
                            dp[i][j][k] = max_prev + current_fruits
        
        # The answer is the maximum fruits collected when all children reach (n-1, n-1)
        return dp[n-1][n-1][n-1]