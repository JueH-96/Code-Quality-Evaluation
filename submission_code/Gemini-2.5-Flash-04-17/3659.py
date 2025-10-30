from typing import List

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        MOD = 10**9 + 7
        XOR_MAX = 16 # Possible XOR values range from 0 to 15 because grid values and k are < 16

        # prev_dp[j][xor_val] stores the number of paths from (0,0) to (i-1, j)
        # with XOR sum xor_val.
        # We only need to store the DP values for the previous row to calculate the current row.
        # The size is n (for columns) x XOR_MAX (for possible XOR values).
        prev_dp = [[0] * XOR_MAX for _ in range(n)]

        # Base case: cell (0, 0). The path starts here, XOR sum is just grid[0][0].
        # There is 1 path to (0, 0) with XOR sum grid[0][0].
        prev_dp[0][grid[0][0]] = 1

        # Fill DP for the rest of the first row (i=0, j=1 to n-1)
        # For row 0, paths can only come from the left cell (0, j-1).
        for j in range(1, n):
            current_cell_val = grid[0][j]
            # For each possible target XOR sum at (0, j)
            for target_xor in range(XOR_MAX):
                 # To reach (0, j) with target_xor, the path to (0, j-1)
                 # must have had an XOR sum `target_xor ^ current_cell_val`.
                 required_prev_xor = target_xor ^ current_cell_val
                 
                 # The number of ways to reach (0, j) with target_xor is
                 # the number of ways to reach (0, j-1) with required_prev_xor.
                 # Use value from prev_dp (which represents row 0).
                 # required_prev_xor will be less than XOR_MAX because both target_xor and current_cell_val are less than XOR_MAX.
                 prev_dp[j][target_xor] = prev_dp[j-1][required_prev_xor]


        # Iterate through the remaining rows (i=1 to m-1)
        for i in range(1, m):
            # curr_dp[j][xor_val] stores the number of paths from (0,0) to (i, j)
            # with XOR sum xor_val. This will store the results for the current row i.
            curr_dp = [[0] * XOR_MAX for _ in range(n)]
            
            # Fill DP for the first cell of the current row (i, j=0)
            # Paths to (i, 0) can only come from the cell above (i-1, 0).
            current_cell_val_first_col = grid[i][0]
            for target_xor in range(XOR_MAX):
                 # To reach (i, 0) with target_xor, the path to (i-1, 0)
                 # must have had XOR sum `target_xor ^ grid[i][0]`.
                 required_prev_xor = target_xor ^ current_cell_val_first_col
                 
                 # The number of ways to reach (i, 0) with target_xor is
                 # the number of ways to reach (i-1, 0) with required_prev_xor.
                 # Use value from prev_dp (which represents row i-1) at column 0
                 # required_prev_xor will be less than XOR_MAX.
                 curr_dp[0][target_xor] = prev_dp[0][required_prev_xor]

            # Fill DP for the rest of the current row (i, j=1 to n-1)
            for j in range(1, n):
                current_cell_val = grid[i][j]
                for target_xor in range(XOR_MAX):
                    # To reach (i, j) with target_xor, the previous cell
                    # (either (i-1, j) or (i, j-1)) must have had an XOR sum
                    # equal to `target_xor ^ grid[i][j]`.
                    required_prev_xor = target_xor ^ current_cell_val

                    # Add paths coming from the cell above (i-1, j)
                    # Use value from prev_dp (which represents row i-1) at column j
                    from_above = prev_dp[j][required_prev_xor]

                    # Add paths coming from the cell to the left (i, j-1)
                    # Use value from curr_dp (which represents current row i) at column j-1
                    from_left = curr_dp[j-1][required_prev_xor]

                    # The total ways to reach (i, j) with target_xor is the sum of ways
                    # from above and from left, modulo MOD.
                    curr_dp[j][target_xor] = (from_above + from_left) % MOD

            # The current row's DP values become the previous row's values for the next iteration.
            prev_dp = curr_dp

        # After iterating through all rows, prev_dp contains the results for row m-1.
        # The final answer is the number of paths to the bottom-right cell (m-1, n-1)
        # with the required XOR sum k. This corresponds to prev_dp[n-1][k].
        result = prev_dp[n - 1][k]

        return result