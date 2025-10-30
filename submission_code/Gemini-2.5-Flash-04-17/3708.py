from typing import List

class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        """
        Traverses a 2D grid in a zigzag pattern, skipping every alternate cell.

        Args:
            grid: An m x n 2D array of positive integers.

        Returns:
            An array of integers containing the values of the visited cells
            in the order they were traversed.
        """
        if not grid or not grid[0]:
            return []

        m = len(grid)
        n = len(grid[0])
        result = []
        traversal_index = 0  # Counts every cell visited in the zigzag path

        for r in range(m):
            if r % 2 == 0:  # Even rows: Traverse right
                for c in range(n):
                    # Visit the cell if its traversal_index is even (0th, 2nd, 4th, ...)
                    if traversal_index % 2 == 0:
                        result.append(grid[r][c])
                    traversal_index += 1
            else:  # Odd rows: Traverse left
                for c in range(n - 1, -1, -1):
                    # Visit the cell if its traversal_index is even (0th, 2nd, 4th, ...)
                    if traversal_index % 2 == 0:
                        result.append(grid[r][c])
                    traversal_index += 1

        return result