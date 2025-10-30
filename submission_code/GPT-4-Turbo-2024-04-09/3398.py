class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        def check_square(x, y):
            """ Check if the 2x2 square starting at (x, y) can be made uniform with at most one change. """
            colors = [grid[x][y], grid[x][y+1], grid[x+1][y], grid[x+1][y+1]]
            count_B = colors.count('B')
            count_W = colors.count('W')
            # If there are 3 or 4 of the same color, we can make it uniform with at most one change
            return count_B >= 3 or count_W >= 3
        
        # Check all possible 2x2 squares in the 3x3 grid
        for i in range(2):
            for j in range(2):
                if check_square(i, j):
                    return True
        return False