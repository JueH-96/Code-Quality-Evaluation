from typing import List

class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        result = []
        rows = len(grid)
        cols = len(grid[0])
        
        for i in range(rows):
            if i % 2 == 0:  # Even row: left to right
                for j in range(0, cols, 2):  # Skip every alternate cell
                    result.append(grid[i][j])
            else:  # Odd row: right to left
                for j in range(cols - 1, -1, -2):  # Skip every alternate cell
                    result.append(grid[i][j])
        
        return result