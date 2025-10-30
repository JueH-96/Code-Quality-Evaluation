class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        
        # Initialize 3D DP table
        dp = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
        
        # Initialize the starting positions
        dp[0][0][n-1] = fruits[0][0] + fruits[0][n-1] + fruits[n-1][0]
        
        # Helper function to check if a position is valid
        def is_valid(x, y):
            return 0 <= x < n and 0 <= y < n
        
        # Iterate through all possible positions
        for sum_coord in range(1, 3*n-3):
            for i in range(n):
                for j in range(n):
                    k = sum_coord - i - j
                    if 0 <= k < n:
                        # Calculate the fruits collected at this step
                        current_fruits = fruits[i][j] + fruits[j][k] + fruits[k][i]
                        if i == j == k:
                            current_fruits = fruits[i][j]
                        elif i == j or j == k or k == i:
                            current_fruits = sum(set([fruits[i][j], fruits[j][k], fruits[k][i]]))
                        
                        # Check all possible previous positions
                        for di, dj, dk in [(0,0,1), (0,1,0), (1,0,0), (0,1,1), (1,0,1), (1,1,0)]:
                            prev_i, prev_j, prev_k = i-di, j-dj, k-dk
                            if is_valid(prev_i, prev_j) and is_valid(prev_j, prev_k) and is_valid(prev_k, prev_i):
                                dp[i][j][k] = max(dp[i][j][k], dp[prev_i][prev_j][prev_k] + current_fruits)
        
        return dp[n-1][n-1][n-1]