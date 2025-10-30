class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])
        
        # dp[i][j][xor_value] will store the number of ways to reach cell (i, j) with XOR value `xor_value`
        dp = [[[0] * 16 for _ in range(n)] for _ in range(m)]
        
        # Initialize the starting point
        dp[0][0][grid[0][0]] = 1
        
        for i in range(m):
            for j in range(n):
                for xor_value in range(16):
                    if dp[i][j][xor_value] > 0:
                        if i + 1 < m:
                            new_xor = xor_value ^ grid[i + 1][j]
                            dp[i + 1][j][new_xor] = (dp[i + 1][j][new_xor] + dp[i][j][xor_value]) % MOD
                        if j + 1 < n:
                            new_xor = xor_value ^ grid[i][j + 1]
                            dp[i][j + 1][new_xor] = (dp[i][j + 1][new_xor] + dp[i][j][xor_value]) % MOD
        
        return dp[m - 1][n - 1][k]

# Example usage:
# sol = Solution()
# print(sol.countPathsWithXorValue([[2, 1, 5], [7, 10, 0], [12, 6, 4]], 11))  # Output: 3
# print(sol.countPathsWithXorValue([[1, 3, 3, 3], [0, 3, 3, 2], [3, 0, 1, 1]], 2))  # Output: 5
# print(sol.countPathsWithXorValue([[1, 1, 1, 2], [3, 0, 3, 2], [3, 0, 2, 2]], 10))  # Output: 0