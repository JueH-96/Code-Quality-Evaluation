class Solution:
    def canMakeSquare(self, grid: list[list[str]]) -> bool:
        for i in range(2):
            for j in range(2):
                square = [grid[i][j], grid[i+1][j], grid[i][j+1], grid[i+1][j+1]]
                if square.count('B') <= 1 or square.count('W') <= 1:
                    return True
        return False