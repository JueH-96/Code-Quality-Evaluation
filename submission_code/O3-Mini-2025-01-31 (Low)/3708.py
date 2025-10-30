from typing import List

class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        # First, perform a full zigzag traversal
        zigzag_order = []
        rows = len(grid)
        for i in range(rows):
            # For even rows, traverse left-to-right;
            # for odd rows, traverse right-to-left.
            if i % 2 == 0:
                zigzag_order.extend(grid[i])
            else:
                zigzag_order.extend(grid[i][::-1])
        # Now, skip every alternate cell (i.e., take cells at even indices)
        result = zigzag_order[0::2]
        return result