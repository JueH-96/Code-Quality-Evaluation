from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        """
        Calculates the number of valid submatrices starting at grid[0][0].
        A submatrix is considered valid if it satisfies three conditions:
        1. Its top-left corner is grid[0][0].
        2. It contains an equal number of 'X's and 'Y's.
        3. It contains at least one 'X'.

        The approach uses 2D prefix sums (integral image) to efficiently calculate
        the counts of 'X's and 'Y's in each submatrix starting from (0,0).

        Args:
            grid: A 2D list of characters where each element is 'X', 'Y', or '.'.
                  Constraints: 1 <= grid.length, grid[i].length <= 1000.

        Returns:
            The total number of submatrices that meet all the specified conditions.
        """
        rows = len(grid)
        # Constraints guarantee rows >= 1, so no need for an explicit empty check
        cols = len(grid[0])
        # Constraints guarantee cols >= 1, so no need for an explicit empty check

        # Initialize prefix sum matrices with dimensions (rows + 1) x (cols + 1).
        # prefixX[r][c] will store the count of 'X's in the submatrix defined by
        # the top-left corner (0,0) and the bottom-right corner (r-1, c-1) in the original grid.
        # Similarly, prefixY[r][c] will store the count of 'Y's.
        # Using +1 dimensions simplifies boundary condition handling in the prefix sum calculation.
        prefixX = [[0] * (cols + 1) for _ in range(rows + 1)]
        prefixY = [[0] * (cols + 1) for _ in range(rows + 1)]

        count = 0 # Initialize the counter for valid submatrices

        # Iterate through each cell (i, j) of the grid. Each cell (i, j) represents
        # the bottom-right corner of a potential submatrix starting at (0, 0).
        for i in range(rows):
            for j in range(cols):
                # Determine if the character at the current cell grid[i][j] is 'X' or 'Y'.
                # Use 1 if it matches, 0 otherwise.
                is_X = 1 if grid[i][j] == 'X' else 0
                is_Y = 1 if grid[i][j] == 'Y' else 0

                # Calculate the prefix sums using the standard 2D recurrence relation:
                # Sum(i, j) = value(i, j) + Sum(i-1, j) + Sum(i, j-1) - Sum(i-1, j-1)
                # Here, Sum(i, j) corresponds to prefixX[i+1][j+1] or prefixY[i+1][j+1]
                # because the prefix arrays are 1-indexed relative to the grid indices (0-based).
                # prefixX[i+1][j+1] stores the total count of 'X's in the submatrix from (0,0) to (i,j).
                prefixX[i+1][j+1] = is_X + prefixX[i][j+1] + prefixX[i+1][j] - prefixX[i][j]
                # prefixY[i+1][j+1] stores the total count of 'Y's in the submatrix from (0,0) to (i,j).
                prefixY[i+1][j+1] = is_Y + prefixY[i][j+1] + prefixY[i+1][j] - prefixY[i][j]

                # Get the counts of 'X' and 'Y' for the submatrix ending at grid[i][j]
                current_X = prefixX[i+1][j+1]
                current_Y = prefixY[i+1][j+1]

                # Check if the current submatrix (defined by top-left (0,0) and bottom-right (i,j)) is valid:
                # Condition 1: Contains grid[0][0] - This is implicitly satisfied by how we define submatrices.
                # Condition 2: Equal frequency of 'X' and 'Y' (current_X == current_Y).
                # Condition 3: At least one 'X' (current_X >= 1).
                # Note: If current_X == current_Y and current_X >= 1, it automatically implies current_Y >= 1.
                if current_X >= 1 and current_X == current_Y:
                    # If all conditions are met, increment the count of valid submatrices.
                    count += 1

        # Return the final count of valid submatrices found.
        return count