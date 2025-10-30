class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        
        # Initialize a 3D DP array to store the maximum fruits collected
        dp = [[[-1] * n for _ in range(n)] for _ in range(n)]
        
        # Base case: All three children start at their respective corners
        dp[0][0][0] = fruits[0][0] + fruits[0][n-1] + fruits[n-1][0]
        
        # Iterate over the number of moves
        for step in range(1, n):
            for i1 in range(max(0, step - (n - 1)), min(n, step + 1)):
                for j1 in range(max(0, step - (n - 1)), min(n, step + 1)):
                    for i2 in range(max(0, step - (n - 1)), min(n, step + 1)):
                        j2 = step - i2
                        if j2 < 0 or j2 >= n:
                            continue
                        
                        # Calculate the maximum fruits collected for the current positions
                        max_fruits = -1
                        for di1, dj1 in [(0, 1), (1, 0), (1, 1)]:
                            for di2, dj2 in [(0, 1), (1, 0), (1, 1)]:
                                ni1, nj1 = i1 - di1, j1 - dj1
                                ni2, nj2 = i2 - di2, j2 - dj2
                                if 0 <= ni1 < n and 0 <= nj1 < n and 0 <= ni2 < n and 0 <= nj2 < n:
                                    if dp[ni1][nj1][ni2] != -1:
                                        max_fruits = max(max_fruits, dp[ni1][nj1][ni2])
                        
                        if max_fruits != -1:
                            dp[i1][j1][i2] = max_fruits + fruits[i1][j1]
                            if i1 != i2 or j1 != j2:
                                dp[i1][j1][i2] += fruits[i2][j2]
        
        # The result is the maximum fruits collected when all three children reach (n-1, n-1)
        return dp[n-1][n-1][n-1]

# Example usage:
# sol = Solution()
# print(sol.maxCollectedFruits([[1,2,3,4],[5,6,8,7],[9,10,11,12],[13,14,15,16]]))  # Output: 100
# print(sol.maxCollectedFruits([[1,1],[1,1]]))  # Output: 4