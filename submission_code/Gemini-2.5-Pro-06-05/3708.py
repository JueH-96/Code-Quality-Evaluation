from typing import List

class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        """
        Traverses a 2D grid in a zigzag pattern, skipping every alternate cell.
        """
        num_rows = len(grid)
        
        # This list will store all cell values in the order of a full zigzag traversal.
        full_zigzag_path = []
        
        # Iterate through each row to build the full traversal path.
        for i in range(num_rows):
            # Even-indexed rows (0, 2, ...) are traversed from left to right.
            if i % 2 == 0:
                full_zigzag_path.extend(grid[i])
            # Odd-indexed rows (1, 3, ...) are traversed from right to left.
            else:
                # grid[i][::-1] creates a reversed copy of the row.
                full_zigzag_path.extend(grid[i][::-1])
                
        # To skip every alternate cell, we take every second element from the
        # generated path, starting from the first element (index 0).
        # Python's slice notation `[::2]` does this efficiently.
        return full_zigzag_path[::2]