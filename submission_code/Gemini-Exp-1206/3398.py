class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        def check_square(grid):
            for i in range(2):
                for j in range(2):
                    if grid[i][j] == grid[i + 1][j] == grid[i][j + 1] == grid[i + 1][j + 1]:
                        return True
            return False

        if check_square(grid):
            return True

        for i in range(3):
            for j in range(3):
                original_color = grid[i][j]
                grid[i][j] = 'W' if original_color == 'B' else 'B'
                if check_square(grid):
                    return True
                grid[i][j] = original_color

        return False