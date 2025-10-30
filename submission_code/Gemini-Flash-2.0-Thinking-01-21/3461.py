class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        min_row = float('inf')
        max_row = float('-inf')
        min_col = float('inf')
        max_col = float('-inf')

        has_one = False
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    has_one = True
                    min_row = min(min_row, i)
                    max_row = max(max_row, i)
                    min_col = min(min_col, j)
                    max_col = max(max_col, j)

        if not has_one:
            return 0

        height = max_row - min_row + 1
        width = max_col - min_col + 1
        return height * width