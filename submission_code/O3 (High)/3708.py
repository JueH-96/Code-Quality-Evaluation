from typing import List

class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        """
        Traverse the 2-D `grid` in a row–wise zig-zag (left→right on even
        rows, right→left on odd rows) while recording every alternate cell,
        starting with the very first cell (0, 0).
        """
        if not grid:                         # defensive programming, though
            return []                        # problem guarantees non-empty

        rows, cols = len(grid), len(grid[0])
        take_next = True                     # whether the current cell is kept
        result = []

        for r in range(rows):
            # determine traversal direction for the current row
            col_range = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)

            for c in col_range:
                if take_next:
                    result.append(grid[r][c])
                take_next = not take_next     # flip for “skip every alternate”

        return result