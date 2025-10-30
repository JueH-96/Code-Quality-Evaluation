from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # There are 4 possible 2x2 squares in a 3x3 grid.
        # We check each square for the number of 'B' and 'W' cells.
        # If either count is 3 or 4, then at most one change is required
        # to make all 4 cells the same color.
        for i in range(2):  # Possible starting rows: 0, 1
            for j in range(2):  # Possible starting columns: 0, 1
                countB = 0
                countW = 0
                # Check the 2x2 block starting at (i, j)
                for di in range(2):
                    for dj in range(2):
                        if grid[i + di][j + dj] == 'B':
                            countB += 1
                        else:
                            countW += 1
                # If either three or four cells are the same,
                # we can flip at most one cell to have a uniform 2x2 block.
                if countB >= 3 or countW >= 3:
                    return True
        return False