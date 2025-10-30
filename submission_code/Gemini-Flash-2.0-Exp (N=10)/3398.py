class Solution:
    def canMakeSquare(self, grid: list[list[str]]) -> bool:
        def check_square(grid):
            for r in range(2):
                for c in range(2):
                    if (grid[r][c] == grid[r][c+1] == grid[r+1][c] == grid[r+1][c+1]):
                        return True
            return False

        if check_square(grid):
            return True

        for r in range(3):
            for c in range(3):
                original_val = grid[r][c]
                grid[r][c] = 'W' if original_val == 'B' else 'B'
                if check_square(grid):
                    return True
                grid[r][c] = original_val
        
        return False