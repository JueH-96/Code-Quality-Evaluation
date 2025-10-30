class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        for i in range(2):
            for j in range(2):
                cells = [grid[i][j], grid[i][j+1], grid[i+1][j], grid[i+1][j+1]]
                count_B = cells.count('B')
                if count_B <= 1 or count_B >= 3:
                    return True
        return False