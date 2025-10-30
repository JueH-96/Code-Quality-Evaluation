class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])
        
        # Initialize a DP table where dp[i][j][x] represents the number of ways to reach (i,j) with XOR value x
        # Since the grid values are up to 15, the XOR values will also be up to 15
        dp = [[dict() for _ in range(n)] for _ in range(m)]
        
        # Start with the initial cell (0,0)
        initial_xor = grid[0][0]
        dp[0][0][initial_xor] = 1
        
        # Iterate through each cell in the grid
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                current_xor = grid[i][j]
                # Initialize the current cell's XOR dictionary
                dp[i][j] = dict()
                # Check the cell above
                if i > 0:
                    for x, cnt in dp[i-1][j].items():
                        new_xor = x ^ current_xor
                        if new_xor in dp[i][j]:
                            dp[i][j][new_xor] = (dp[i][j][new_xor] + cnt) % MOD
                        else:
                            dp[i][j][new_xor] = cnt % MOD
                # Check the cell to the left
                if j > 0:
                    for x, cnt in dp[i][j-1].items():
                        new_xor = x ^ current_xor
                        if new_xor in dp[i][j]:
                            dp[i][j][new_xor] = (dp[i][j][new_xor] + cnt) % MOD
                        else:
                            dp[i][j][new_xor] = cnt % MOD
        
        # The answer is the number of ways to reach (m-1, n-1) with XOR value k
        if k in dp[m-1][n-1]:
            return dp[m-1][n-1][k] % MOD
        else:
            return 0