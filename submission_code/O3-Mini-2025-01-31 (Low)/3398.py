class Solution:
    def canMakeSquare(self, grid: list[list[str]]) -> bool:
        # Check each possible 2x2 subgrid in the 3x3 grid.
        for i in range(2):
            for j in range(2):
                # Extract the 2x2 square cells
                cells = [grid[i][j], grid[i][j+1], grid[i+1][j], grid[i+1][j+1]]
                # Count the occurrences of 'B' and 'W'
                countB = cells.count('B')
                countW = cells.count('W')
                # If at least three cells are the same, we can change one cell (if needed)
                if countB >= 3 or countW >= 3:
                    return True
        return False