class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Check all four possible 2x2 squares in the 3x3 grid
        for i in range(2):
            for j in range(2):
                # Extract the four cells of the current 2x2 square
                cells = [
                    grid[i][j],
                    grid[i][j+1],
                    grid[i+1][j],
                    grid[i+1][j+1]
                ]
                # Calculate the number of cells not 'B' and not 'W'
                non_b = sum(1 for cell in cells if cell != 'B')
                non_w = sum(1 for cell in cells if cell != 'W')
                # Check if either color can be achieved with at most one change
                if non_b <= 1 or non_w <= 1:
                    return True
        # If none of the squares can be fixed with at most one change
        return False