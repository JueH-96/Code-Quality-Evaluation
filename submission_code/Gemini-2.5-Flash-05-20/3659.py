from typing import List

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m = len(grid)
        n = len(grid[0])

        # dp[i][j][xor_val] stores the number of paths from (0,0) to (i,j)
        # such that the XOR sum of elements along the path is xor_val.
        # The maximum possible XOR sum is 15 (since grid values and k are < 16,
        # XORing 4-bit numbers results in a 4-bit number).
        # So, the third dimension size is 16 (for XOR values 0-15).
        dp = [[[0] * 16 for _ in range(n)] for _ in range(m)]

        # Base case: For the starting cell (0, 0), there is one path to itself,
        # and its XOR sum is just the value of grid[0][0].
        dp[0][0][grid[0][0]] = 1

        # Iterate through the grid cells row by row, then column by column
        for i in range(m):
            for j in range(n):
                current_cell_val = grid[i][j]

                # If it's the starting cell (0, 0), its count is already initialized.
                # It cannot be reached from other cells, so skip further calculations for it.
                if i == 0 and j == 0:
                    continue

                # Iterate through all possible XOR sums (0 to 15) that could have been
                # accumulated up to the previous cell (either above or to the left).
                for prev_xor_sum in range(16):
                    # Case 1: Coming from the cell directly above (i-1, j)
                    if i > 0:
                        # If there are paths to (i-1, j) with `prev_xor_sum`
                        if dp[i-1][j][prev_xor_sum] > 0:
                            # Calculate the new XOR sum for the path ending at (i, j)
                            # by XORing the `prev_xor_sum` with the `current_cell_val`.
                            new_xor_sum = prev_xor_sum ^ current_cell_val
                            # Add the count of paths from (i-1, j) to the current cell's state
                            dp[i][j][new_xor_sum] = (dp[i][j][new_xor_sum] + dp[i-1][j][prev_xor_sum]) % MOD
                    
                    # Case 2: Coming from the cell directly to the left (i, j-1)
                    if j > 0:
                        # If there are paths to (i, j-1) with `prev_xor_sum`
                        if dp[i][j-1][prev_xor_sum] > 0:
                            # Calculate the new XOR sum for the path ending at (i, j)
                            new_xor_sum = prev_xor_sum ^ current_cell_val
                            # Add the count of paths from (i, j-1) to the current cell's state
                            dp[i][j][new_xor_sum] = (dp[i][j][new_xor_sum] + dp[i][j-1][prev_xor_sum]) % MOD
        
        # The final answer is the number of paths to the bottom-right cell (m-1, n-1)
        # whose total XOR sum is equal to `k`.
        return dp[m-1][n-1][k]