from typing import List

class Solution:
  def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
    m = len(grid)
    n = len(grid[0])
    MOD = 10**9 + 7
    
    # Values in grid and k are < 16.
    # So, all XOR sums will be < 16 (i.e., in the range [0, 15]).
    # We need to store counts for XOR sums 0, 1, ..., 15. So, size is 16.
    MAX_XOR_VAL = 16 

    # dp[r][c][xor_s] = number of paths to (r,c) where the
    # XOR sum of elements along the path (inclusive of grid[r][c]) is xor_s.
    dp = [[[0] * MAX_XOR_VAL for _ in range(n)] for _ in range(m)]

    # Base case: For the starting cell (0,0)
    # The path consists of only grid[0][0]. Its XOR sum is grid[0][0].
    dp[0][0][grid[0][0]] = 1

    # Fill the first row: Paths can only come from the left.
    # For cell (0, c_idx), the path comes from (0, c_idx-1).
    for c_idx in range(1, n):
        current_cell_val = grid[0][c_idx]
        for prev_xor_sum in range(MAX_XOR_VAL):
            count_from_left_cell = dp[0][c_idx-1][prev_xor_sum]
            if count_from_left_cell > 0:
                new_xor_sum = prev_xor_sum ^ current_cell_val
                dp[0][c_idx][new_xor_sum] = (dp[0][c_idx][new_xor_sum] + count_from_left_cell) % MOD
    
    # Fill the first column: Paths can only come from above.
    # For cell (r_idx, 0), the path comes from (r_idx-1, 0).
    for r_idx in range(1, m):
        current_cell_val = grid[r_idx][0]
        for prev_xor_sum in range(MAX_XOR_VAL):
            count_from_up_cell = dp[r_idx-1][0][prev_xor_sum]
            if count_from_up_cell > 0:
                new_xor_sum = prev_xor_sum ^ current_cell_val
                dp[r_idx][0][new_xor_sum] = (dp[r_idx][0][new_xor_sum] + count_from_up_cell) % MOD

    # Fill the rest of the DP table.
    # For cell (r_idx, c_idx), paths can come from (r_idx-1, c_idx) (up) or (r_idx, c_idx-1) (left).
    for r_idx in range(1, m):
        for c_idx in range(1, n):
            current_cell_val = grid[r_idx][c_idx]
            for prev_xor_sum in range(MAX_XOR_VAL):
                # Contribution from cell (r_idx-1, c_idx) (cell above)
                count_from_up_cell = dp[r_idx-1][c_idx][prev_xor_sum]
                if count_from_up_cell > 0:
                    new_xor_sum = prev_xor_sum ^ current_cell_val
                    dp[r_idx][c_idx][new_xor_sum] = (dp[r_idx][c_idx][new_xor_sum] + count_from_up_cell) % MOD
                
                # Contribution from cell (r_idx, c_idx-1) (cell to the left)
                count_from_left_cell = dp[r_idx][c_idx-1][prev_xor_sum]
                if count_from_left_cell > 0:
                    new_xor_sum = prev_xor_sum ^ current_cell_val
                    dp[r_idx][c_idx][new_xor_sum] = (dp[r_idx][c_idx][new_xor_sum] + count_from_left_cell) % MOD
                    
    # The result is the number of paths to (m-1, n-1) with XOR sum k.
    return dp[m-1][n-1][k]