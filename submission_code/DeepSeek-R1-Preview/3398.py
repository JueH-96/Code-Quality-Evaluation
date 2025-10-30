class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Check all four 2x2 squares
        squares = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for i, j in squares:
            cells = [
                grid[i][j],
                grid[i][j+1],
                grid[i+1][j],
                grid[i+1][j+1]
            ]
            count_B = sum(1 for c in cells if c == 'B')
            count_W = 4 - count_B
            if count_B >= 3 or count_W >= 3:
                return True
        return False