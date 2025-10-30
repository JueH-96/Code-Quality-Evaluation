class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        def check(grid, color):
            for i in range(2):
                for j in range(2):
                    if grid[i][j] != color:
                        return False
            return True

        for i in range(3):
            for j in range(3):
                original = grid[i][j]
                grid[i][j] = 'W'
                if check(grid, 'W'):
                    return True
                grid[i][j] = 'B'
                if check(grid, 'B'):
                    return True
                grid[i][j] = original
        return False