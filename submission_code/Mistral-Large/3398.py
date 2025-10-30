from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        def check_square(x, y):
            return (grid[x][y] == grid[x+1][y] == grid[x][y+1] == grid[x+1][y+1])

        def one_away_square(x, y):
            colors = [grid[x][y], grid[x+1][y], grid[x][y+1], grid[x+1][y+1]]
            return colors.count('W') == 1 or colors.count('B') == 1

        for i in range(2):
            for j in range(2):
                if check_square(i, j):
                    return True
                if one_away_square(i, j):
                    return True

        return False