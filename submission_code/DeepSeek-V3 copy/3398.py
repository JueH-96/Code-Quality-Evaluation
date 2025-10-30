class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Iterate over all possible 2x2 squares in the 3x3 grid
        for i in range(2):
            for j in range(2):
                # Get the four cells in the current 2x2 square
                cells = [grid[i][j], grid[i][j+1], grid[i+1][j], grid[i+1][j+1]]
                # Count the number of 'B's and 'W's
                count_B = cells.count('B')
                count_W = cells.count('W')
                # If either count is 3 or 4, it's possible to make a uniform square
                if count_B >= 3 or count_W >= 3:
                    return True
        return False