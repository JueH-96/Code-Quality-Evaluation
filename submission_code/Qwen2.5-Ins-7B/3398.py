from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        for i in range(2):
            for j in range(2):
                if grid[i][j] == grid[i][j+1] == grid[i+1][j] == grid[i+1][j+1]:
                    return True
                if (i == 0 and grid[i][j] == grid[i][j+1] == grid[i+1][j]) or \
                   (i == 0 and grid[i][j] == grid[i+1][j] == grid[i+1][j+1]) or \
                   (i == 1 and grid[i][j] == grid[i][j+1] == grid[i-1][j+1]) or \
                   (i == 1 and grid[i][j] == grid[i-1][j] == grid[i-1][j+1]):
                    return True
        return False