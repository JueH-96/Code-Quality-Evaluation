class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        for i in range(2):
            for j in range(2):
                # Extract the four cells of the current 2x2 square
                cells = [
                    grid[i][j],
                    grid[i][j+1],
                    grid[i+1][j],
                    grid[i+1][j+1]
                ]
                # Check for both possible target colors
                for target in ['B', 'W']:
                    # Calculate the number of changes needed
                    changes = sum(1 for c in cells if c != target)
                    if changes <= 1:
                        return True
        # If no square can be made uniform with at most one change
        return False