class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        row_counts = [0] * rows
        col_counts = [0] * cols

        # Count the number of 1s in each row and column
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    row_counts[i] += 1
                    col_counts[j] += 1

        # Calculate the number of right triangles
        count = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    # For each 1, the number of right triangles it can form is the product of the number of 1s in its row and column
                    count += (row_counts[i] - 1) * (col_counts[j] - 1)

        return count