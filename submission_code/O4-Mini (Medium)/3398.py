from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # There are only four possible 2x2 sub-squares in a 3x3 grid:
        # top-left corners at (0,0), (0,1), (1,0), and (1,1).
        for i in range(2):        # i = 0 or 1
            for j in range(2):    # j = 0 or 1
                countW = 0
                # Count whites in the 2x2 block with top-left (i,j)
                for di in (0, 1):
                    for dj in (0, 1):
                        if grid[i + di][j + dj] == 'W':
                            countW += 1
                countB = 4 - countW
                # If at most one cell is of the wrong color, we can flip it
                if countW <= 1 or countB <= 1:
                    return True
        return False