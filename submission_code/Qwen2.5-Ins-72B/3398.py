class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        def is_square(x, y, color):
            return grid[x][y] == color and grid[x][y + 1] == color and grid[x + 1][y] == color and grid[x + 1][y + 1] == color
        
        def can_form_square(x, y):
            if is_square(x, y, grid[x][y]):
                return True
            for i in range(x, x + 2):
                for j in range(y, y + 2):
                    if grid[i][j] != grid[x][y]:
                        grid[i][j] = grid[x][y]
                        if is_square(x, y, grid[x][y]):
                            return True
                        grid[i][j] = 'W' if grid[x][y] == 'B' else 'B'
            return False
        
        for i in range(2):
            for j in range(2):
                if can_form_square(i, j):
                    return True
        return False