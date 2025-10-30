class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        for i in range(2):
            for j in range(2):
                cells = [
                    grid[i][j],
                    grid[i][j+1],
                    grid[i+1][j],
                    grid[i+1][j+1]
                ]
                count_b = cells.count('B')
                count_w = cells.count('W')
                if max(count_b, count_w) >= 3:
                    return True
        return False