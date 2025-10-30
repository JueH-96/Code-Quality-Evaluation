from typing import List

class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        zigzag = []
        
        # Traverse each row in zigzag order.
        for i, row in enumerate(grid):
            if i % 2 == 0:
                # For even-indexed rows, traverse left to right.
                zigzag.extend(row)
            else:
                # For odd-indexed rows, traverse right to left.
                zigzag.extend(row[::-1])
                
        # Skip every alternate cell globally (take elements at indices 0, 2, 4, ...)
        result = zigzag[::2]
        return result