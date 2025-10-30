from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Check all possible 2x2 subgrids
        for i in range(2):
            for j in range(2):
                # Count the number of 'B' and 'W' in the current 2x2 subgrid
                count_B = 0
                count_W = 0
                for x in range(i, i + 2):
                    for y in range(j, j + 2):
                        if grid[x][y] == 'B':
                            count_B += 1
                        else:
                            count_W += 1
                # If the subgrid already has 4 of the same color, return True
                if count_B == 4 or count_W == 4:
                    return True
                # If the subgrid has 3 of the same color, we can change the remaining one to match
                if count_B == 3 or count_W == 3:
                    return True
        return False