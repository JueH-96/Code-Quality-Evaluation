from typing import List

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        """
        Calculates the number of paths from (0,0) to (m-1, n-1) such that
        the XOR sum of elements on the path equals k.

        The solution uses dynamic programming.
        Let dp[i][j][val] be the number of paths from (0,0) to (i,j)
        with a path XOR sum of `val`.
        """
        m = len(grid)
        n = len(grid[0])
        MOD = 10**9 + 7

        # Since grid values and k are < 16, the max XOR sum will also be < 16.
        # The possible XOR sums are in the range [0, 15].
        # We need a size of 16 to store counts for these sums.
        max_xor_val = 16

        # dp[i][j][val] = number of paths to (i, j) with XOR sum `val`.
        dp = [[[0] * max_xor_val for _ in range(n)] for _ in range(m)]

        # Base case: At the starting cell (0, 0), there's one path
        # consisting of just that cell. The XOR sum is its value.
        dp[0][0][grid[0][0]] = 1

        # Fill the DP table by iterating through each cell.
        for i in range(m):
            for j in range(n):
                # The base case is already handled.
                if i == 0 and j == 0:
                    continue

                current_val = grid[i][j]
                
                # For each possible XOR sum `x` at the current cell (i, j)
                for x in range(max_xor_val):
                    # To have a path with XOR sum `x` at (i,j), the path to the
                    # previous cell must have an XOR sum of `prev_xor`.
                    # prev_xor ^ current_val = x  => prev_xor = x ^ current_val
                    prev_xor = x ^ current_val

                    # Number of paths from the cell above (if it exists)
                    from_top = dp[i - 1][j][prev_xor] if i > 0 else 0
                    
                    # Number of paths from the cell to the left (if it exists)
                    from_left = dp[i][j - 1][prev_xor] if j > 0 else 0
                    
                    # Sum the counts and take modulo
                    dp[i][j][x] = (from_top + from_left) % MOD

        # The result is the number of paths to the bottom-right cell (m-1, n-1)
        # with a total XOR sum of k.
        return dp[m - 1][n - 1][k]