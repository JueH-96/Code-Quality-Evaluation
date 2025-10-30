from typing import List

class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        # According to the problem description and examples:
        # The grid is an m x n array, where m is the number of rows and n is the number of columns.
        # The constraints section confusingly uses n for rows and m for columns, but the examples
        # and standard matrix notation suggest len(grid) is rows and len(grid[0]) is columns.
        # We will follow the examples and standard notation (m rows x n columns).
        # Let's use clearer variable names: num_rows and num_cols.
        num_rows = len(grid)
        if num_rows == 0:
            return []
        num_cols = len(grid[0])
        if num_cols == 0:
            return []

        result = []
        # Keep track of the index of the current cell in the *full* zigzag traversal sequence.
        # We want to pick elements at indices 0, 2, 4, ... in this sequence.
        traversal_index_count = 0

        # Traverse row by row
        for r in range(num_rows):
            # For even-indexed rows (0, 2, 4, ...), traverse left to right
            if r % 2 == 0:
                for c in range(num_cols):
                    # If the current cell's index in the full traversal sequence is even, add it to the result.
                    if traversal_index_count % 2 == 0:
                        result.append(grid[r][c])
                    # Increment the counter for *every* cell visited in the full traversal path,
                    # regardless of whether it was added to the result.
                    traversal_index_count += 1
            # For odd-indexed rows (1, 3, 5, ...), traverse right to left
            else:
                for c in range(num_cols - 1, -1, -1):
                    # If the current cell's index in the full traversal sequence is even, add it to the result.
                    if traversal_index_count % 2 == 0:
                        result.append(grid[r][c])
                    # Increment the counter for *every* cell visited in the full traversal path.
                    traversal_index_count += 1

        return result