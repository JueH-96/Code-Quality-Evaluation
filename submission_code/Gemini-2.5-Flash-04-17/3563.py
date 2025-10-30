from typing import List
import functools

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        """
        Calculates the maximum score by selecting cells from a grid.

        Conditions:
        1. Select one or more cells.
        2. No two selected cells are in the same row.
        3. Values in the set of selected cells are unique.
        4. Score is the sum of values.

        Args:
            grid: A 2D matrix of positive integers.

        Returns:
            The maximum score achievable.
        """
        m = len(grid)
        n = len(grid[0])

        # DP state: (row_idx, used_mask)
        # row_idx: the index of the current row being considered (from 0 to m)
        # used_mask: an integer bitmask where the k-th bit is set if value k has been used
        # in a selection from rows 0 to row_idx-1. Values are 1 to 100.
        # We need bits 1 through 100 in the mask. The 0th bit is unused.

        # Use functools.lru_cache for memoization. The state (row_idx, used_mask) is hashable.
        # Setting maxsize=None means the cache can grow indefinitely (limited by memory).
        @functools.lru_cache(None)
        def solve(row_idx: int, used_mask: int) -> int:
            """
            Recursive helper function to find the maximum score.

            Args:
                row_idx: The current row index being considered.
                used_mask: Bitmask representing values already used in previous rows.

            Returns:
                The maximum score achievable from row_idx onwards given the used_mask.
            """
            # Base case: If we have finished processing all rows (reached beyond the last row)
            # no more cells can be selected, so the additional score from here is 0.
            if row_idx == m:
                return 0

            # Option 1: Don't select any cell from the current row (row_idx)
            # Get the maximum score from the remaining rows (row_idx + 1 onwards)
            # without using any value from the current row. The set of used values remains the same.
            max_score = solve(row_idx + 1, used_mask)

            # Option 2: Select a cell from the current row (row_idx)
            # Iterate through all possible columns in the current row to find a cell to select.
            for col_idx in range(n):
                value = grid[row_idx][col_idx]
                
                # Check if the value of the current cell has been used before.
                # The bit corresponding to value 'v' is '1 << v'.
                # We check if the 'value'-th bit is NOT set in the 'used_mask'.
                # This is equivalent to checking if the bitwise AND is 0.
                # Note: values are 1-100, so we shift by 'value', using bits 1 through 100.
                # Since values are positive (>= 1) and <= 100, 1 << value is safe and uses appropriate bits.
                if not (used_mask & (1 << value)):
                    # If the value is not used, we can potentially select this cell.
                    
                    # Create a new mask by setting the bit corresponding to the selected value.
                    # This new mask represents the set of used values including the current one.
                    new_mask = used_mask | (1 << value)
                    
                    # Calculate the potential score if we select this cell:
                    # It's the current cell's value plus the maximum score obtainable
                    # from the next row onwards with the updated set of used values.
                    score_if_selected = value + solve(row_idx + 1, new_mask)
                    
                    # Update the maximum score achievable from the current state (row_idx, used_mask).
                    # We take the maximum between the score obtained by not picking from this row
                    # (which is 'max_score' from Option 1) and the score obtained by picking this
                    # specific cell (from Option 2).
                    max_score = max(max_score, score_if_selected)

            # Return the maximum score found for this state (row_idx, used_mask).
            return max_score

        # Initial call to the recursive function:
        # Start considering from the first row (row_idx = 0) with an empty set of used values (used_mask = 0).
        # The DP calculates the maximum score achievable across all valid selections, including the empty selection (score 0).
        # The problem requires selecting one or more cells. Since all grid values are positive (1 <= grid[i][j] <= 100),
        # any valid selection of one or more cells will result in a positive total score.
        # As the constraints guarantee 1 <= grid.length and 1 <= grid[i].length, the grid is non-empty,
        # and a valid selection of at least one cell is always possible (e.g., pick the first cell grid[0][0] and nothing else).
        # This ensures that the maximum score achievable for selecting one or more cells is positive.
        # The DP result (max_score including the empty selection) will thus be equal to the maximum score
        # for selecting one or more cells, because the score for the empty selection (0) will be less than
        # any positive score achievable by selecting one or more cells.
        result = solve(0, 0)

        return result