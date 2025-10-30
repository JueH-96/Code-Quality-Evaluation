class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        rows = []
        cols = []
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 1:
                    rows.append(i)
                    cols.append(j)
        min_r = min(rows)
        max_r = max(rows)
        min_c = min(cols)
        max_c = max(cols)
        return (max_r - min_r + 1) * (max_c - min_c + 1)