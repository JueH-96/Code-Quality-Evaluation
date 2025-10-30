from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        R = len(grid)
        C = len(grid[0])

        # Initialize 2D prefix sum arrays for 'X' and 'Y' counts.
        # prefix_x_counts[r][c] will store the number of 'X's in the submatrix
        # with top-left (0,0) and bottom-right (r,c).
        prefix_x_counts = [[0] * C for _ in range(R)]
        prefix_y_counts = [[0] * C for _ in range(R)]

        ans = 0

        # Iterate through each cell (r, c) of the grid.
        # Each (r, c) represents the bottom-right corner of a submatrix
        # whose top-left corner is fixed at (0,0).
        for r in range(R):
            for c in range(C):
                # Determine the contribution of the current cell grid[r][c]
                # to the 'X' and 'Y' counts.
                val_x = 0
                val_y = 0
                if grid[r][c] == 'X':
                    val_x = 1
                elif grid[r][c] == 'Y':
                    val_y = 1

                # Calculate the total 'X' and 'Y' counts for the submatrix (0,0) to (r,c)
                # using the 2D prefix sum formula:
                # S[r][c] = S[r-1][c] + S[r][c-1] - S[r-1][c-1] + grid_value[r][c]
                
                # Start with the current cell's value
                current_x_sum = val_x
                current_y_sum = val_y

                # Add counts from the submatrix directly above (if available)
                if r > 0:
                    current_x_sum += prefix_x_counts[r-1][c]
                    current_y_sum += prefix_y_counts[r-1][c]
                
                # Add counts from the submatrix directly to the left (if available)
                if c > 0:
                    current_x_sum += prefix_x_counts[r][c-1]
                    current_y_sum += prefix_y_counts[r][c-1]
                
                # Subtract counts from the overlapping top-left submatrix (if available)
                # to avoid double-counting
                if r > 0 and c > 0:
                    current_x_sum -= prefix_x_counts[r-1][c-1]
                    current_y_sum -= prefix_y_counts[r-1][c-1]

                # Store the computed prefix sums for the current cell (r,c)
                prefix_x_counts[r][c] = current_x_sum
                prefix_y_counts[r][c] = current_y_sum

                # Check the problem's conditions for the current submatrix:
                # 1. Equal frequency of 'X' and 'Y' (current_x_sum == current_y_sum)
                # 2. At least one 'X' (current_x_sum > 0)
                # The first problem condition "contains grid[0][0]" is inherently satisfied
                # because all considered submatrices start at (0,0).
                if current_x_sum == current_y_sum and current_x_sum > 0:
                    ans += 1
        
        return ans