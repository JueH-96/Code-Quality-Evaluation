class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        for i in range(2):
            for j in range(2):
                # Extract the 2x2 square
                colors = [grid[i][j], grid[i][j+1],
                          grid[i+1][j], grid[i+1][j+1]]
                # Check if changing at most one cell can make all colors the same
                for target_color in ['B', 'W']:
                    differences = sum(c != target_color for c in colors)
                    if differences <= 1:
                        return True
        return False