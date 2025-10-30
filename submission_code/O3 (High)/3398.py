from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # There are only four possible 2x2 squares in a 3x3 grid
        for r in range(2):          # rows 0 and 1 are the top of a 2-row window
            for c in range(2):      # cols 0 and 1 are the left of a 2-col window
                # Count how many cells in this 2x2 square are 'B'
                black = 0
                for dr in range(2):
                    for dc in range(2):
                        if grid[r+dr][c+dc] == 'B':
                            black += 1
                # A 2x2 square can be made uniform with ≤1 change
                # iff it currently has 0, 1, 3 or 4 cells of the same colour.
                # (0/4 → already uniform; 1/3 → flip the single minority cell)
                if black in (0, 1, 3, 4):
                    return True
        return False