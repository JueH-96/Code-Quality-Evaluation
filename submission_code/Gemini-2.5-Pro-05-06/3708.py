from typing import List

class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        num_rows = len(grid)
        # Constraints: 2 <= num_rows (grid.length) <= 50. So grid is not empty.
        num_cols = len(grid[0])
        # Constraints: 2 <= num_cols (grid[i].length) <= 50. So rows are not empty.

        result = []
        # element_path_index tracks the 0-based index of cells in the full zigzag path.
        # We select cells whose index in this path is even (0th, 2nd, 4th, ...).
        element_path_index = 0 

        for r in range(num_rows):
            if r % 2 == 0:  # Even-indexed row (0, 2, ...): Traverse Left to Right
                for c in range(num_cols):
                    cell_value = grid[r][c]
                    # If it's an element at an even position in the zigzag path, add it.
                    if element_path_index % 2 == 0: 
                        result.append(cell_value)
                    element_path_index += 1  # Increment for every cell encountered
            else:  # Odd-indexed row (1, 3, ...): Traverse Right to Left
                for c in range(num_cols - 1, -1, -1): # Iterate from last col down to first col
                    cell_value = grid[r][c]
                    # If it's an element at an even position in the zigzag path, add it.
                    if element_path_index % 2 == 0:
                        result.append(cell_value)
                    element_path_index += 1  # Increment for every cell encountered
        
        return result