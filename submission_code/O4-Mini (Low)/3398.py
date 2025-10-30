from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # There are four possible 2x2 squares in a 3x3 grid:
        # top-left at (0,0), (0,1), (1,0), and (1,1).
        for i in range(2):
            for j in range(2):
                # Count 'B' and 'W' in the 2x2 square starting at (i, j)
                blacks = 0
                whites = 0
                for di in (0, 1):
                    for dj in (0, 1):
                        if grid[i + di][j + dj] == 'B':
                            blacks += 1
                        else:
                            whites += 1
                # If we already have 4 of one color or can flip one to make 4:
                if blacks >= 3 or whites >= 3:
                    return True
        return False