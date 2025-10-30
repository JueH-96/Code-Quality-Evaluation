class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        def check_square(x, y, color):
            for i in range(x, x + 2):
                for j in range(y, y + 2):
                    if grid[i][j] != color:
                        return False
            return True

        for i in range(2):
            for j in range(2):
                if check_square(i, j, 'B') or check_square(i, j, 'W'):
                    return True

        for i in range(3):
            for j in range(3):
                original_color = grid[i][j]
                grid[i][j] = 'B' if original_color == 'W' else 'W'
                for x in range(2):
                    for y in range(2):
                        if check_square(x, y, 'B') or check_square(x, y, 'W'):
                            return True
                grid[i][j] = original_color

        return False