class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        # Initialize the variables to track the rectangle boundaries
        top, bottom, left, right = float('inf'), float('-inf'), float('inf'), float('-inf')
        
        # Iterate over the grid to find the boundaries
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    top = min(top, i)
                    bottom = max(bottom, i)
                    left = min(left, j)
                    right = max(right, j)
        
        # Calculate the area of the rectangle
        height = bottom - top + 1
        width = right - left + 1
        return height * width