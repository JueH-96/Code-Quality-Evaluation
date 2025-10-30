import collections # This import is not strictly needed but doesn't hurt. Let's remove it for cleaner code.
from typing import List

class Solution:
  """
  This class provides a solution to count paths in a grid with a specific XOR sum,
  moving only right or down.
  """
  def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
    """
    Calculates the number of paths from the top-left cell (0, 0) to the 
    bottom-right cell (m - 1, n - 1) in a grid such that the XOR sum of 
    the elements along the path equals k. Paths can only move right or down.

    Args:
      grid: A 2D list of integers representing the grid. Values are between 0 and 15.
      k: The target XOR sum, between 0 and 15.

    Returns:
      The total number of such paths modulo 10^9 + 7.
    """
    MOD = 10**9 + 7
    m = len(grid)
    n = len(grid[0])

    # dp[i][j][x] stores the number of paths from (0, 0) to (i, j)
    # such that the XOR sum of elements along the path is x.
    # The dimensions are m x n x 16.
    # Since grid values and k are less than 16 (implying they fit in 4 bits),
    # the XOR sum of any path will also be less than 16.
    # We only need to track XOR sums from 0 to 15.
    dp = [[[0] * 16 for _ in range(n)] for _ in range(m)]

    # Base case: Initialize the starting cell (0, 0).
    # The only path to (0, 0) starts and ends there.
    # Its XOR sum is simply the value of the cell grid[0][0].
    # Since grid[0][0] < 16, this index is valid.
    dp[0][0][grid[0][0]] = 1

    # Fill the DP table iteratively using dynamic programming.
    # Iterate through each cell of the grid.
    for i in range(m):
      for j in range(n):
        # Skip the base case cell (0, 0) because it's already initialized
        # and its calculation depends on non-existent previous cells.
        if i == 0 and j == 0:
          continue

        current_val = grid[i][j]
        
        # Iterate through all possible XOR sums (0 to 15) that could be achieved at cell (i, j).
        for current_xor_sum in range(16):
          # To reach cell (i, j) with a path having XOR sum `current_xor_sum`,
          # the path must have come from either the cell above (i-1, j) or the cell to the left (i, j-1).
          
          # If the path came from a previous cell with XOR sum `prev_xor_sum`,
          # then `prev_xor_sum ^ current_val = current_xor_sum`.
          # We can find the required previous XOR sum using the property of XOR:
          # `prev_xor_sum = current_xor_sum ^ current_val`.
          prev_xor_sum = current_xor_sum ^ current_val

          # Initialize the count for dp[i][j][current_xor_sum]. This represents the number of ways
          # to reach (i, j) with the target `current_xor_sum`.
          count = 0

          # Add paths coming from the cell above (i-1, j), if it exists (i.e., i > 0).
          # The number of such paths ending at (i-1, j) with XOR sum `prev_xor_sum` is dp[i-1][j][prev_xor_sum].
          if i > 0:
            count = (count + dp[i - 1][j][prev_xor_sum]) % MOD

          # Add paths coming from the cell to the left (i, j-1), if it exists (i.e., j > 0).
          # The number of such paths ending at (i, j-1) with XOR sum `prev_xor_sum` is dp[i][j-1][prev_xor_sum].
          if j > 0:
            count = (count + dp[i][j - 1][prev_xor_sum]) % MOD
          
          # Store the total count in the DP table for cell (i, j) and XOR sum `current_xor_sum`.
          dp[i][j][current_xor_sum] = count

    # The final answer is the number of paths reaching the bottom-right cell (m-1, n-1)
    # with a total XOR sum equal to the target value k.
    return dp[m - 1][n - 1][k]