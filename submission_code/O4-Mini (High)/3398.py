from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # We only need to check the 4 possible 2x2 sub-squares in the 3x3 grid.
        # For each 2x2 block, if we have at least 3 cells of the same color,
        # we can flip at most one to make all 4 the same.
        
        for i in range(2):      # i = 0 or 1
            for j in range(2):  # j = 0 or 1
                # Count how many are 'B' in this 2x2 block
                countB = 0
                for di in (0, 1):
                    for dj in (0, 1):
                        if grid[i+di][j+dj] == 'B':
                            countB += 1
                countW = 4 - countB
                # If either color already appears at least 3 times,
                # we can flip the other (at most one) to make a uniform 2x2.
                if countB >= 3 or countW >= 3:
                    return True
        
        return False