class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        rows, cols = set(), set()
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                if val:
                    rows.add(i)
                    cols.add(j)
        return (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)