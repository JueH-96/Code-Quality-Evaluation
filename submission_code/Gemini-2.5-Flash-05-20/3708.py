from typing import List

class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        m = len(grid)  # Number of rows
        n = len(grid[0]) # Number of columns

        result = []
        cell_count = 0 # Keeps track of the total number of cells encountered in zigzag path

        # Iterate through each row of the grid
        for r in range(m):
            if r % 2 == 0:
                # If the row index is even, traverse from left to right (columns 0 to n-1)
                for c in range(n):
                    cell_count += 1
                    # If the current cell is the 1st, 3rd, 5th, etc. in the overall traversal order
                    if cell_count % 2 == 1: 
                        result.append(grid[r][c])
            else:
                # If the row index is odd, traverse from right to left (columns n-1 down to 0)
                for c in range(n - 1, -1, -1): # range(start, stop, step)
                    cell_count += 1
                    # If the current cell is the 1st, 3rd, 5th, etc. in the overall traversal order
                    if cell_count % 2 == 1: 
                        result.append(grid[r][c])
        
        return result