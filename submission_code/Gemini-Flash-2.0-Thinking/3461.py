class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        min_row, max_row = rows, -1
        min_col, max_col = cols, -1

        found_one = False
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    found_one = True
                    min_row = min(min_row, r)
                    max_row = max(max_row, r)
                    min_col = min(min_col, c)
                    max_col = max(max_col, c)

        if not found_one:
            return 0  # Should not happen based on constraints

        height = max_row - min_row + 1
        width = max_col - min_col + 1
        return height * width