class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        from collections import defaultdict

        # Get the dimensions of the grid
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        # Initialize counters for 1s in each row and column
        row_count = [0] * rows
        col_count = [0] * cols

        # Count the number of 1s in each row and column
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    row_count[r] += 1
                    col_count[c] += 1

        # Calculate the number of right triangles
        right_triangles = 0

        # Iterate over each cell in the grid
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    # For each 1, calculate the possible triangles
                    # by choosing two other 1s: one in the same row and one in the same column
                    # Subtract 1 to exclude the current cell itself
                    right_triangles += (row_count[r] - 1) * (col_count[c] - 1)

        return right_triangles