class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])

        # Find the range of rows and columns that contain 1s
        min_row, max_row = rows, 0
        min_col, max_col = cols, 0

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    min_row = min(min_row, r)
                    max_row = max(max_row, r)
                    min_col = min(min_col, c)
                    max_col = max(max_col, c)

        # Calculate the dimensions of the smallest rectangle
        height = max_row - min_row + 1
        width = max_col - min_col + 1

        return height * width