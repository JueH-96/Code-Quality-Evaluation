class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        cells = []
        n = len(grid)
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                cells.append((grid[r][c], r))
        cells.sort(reverse=True)
        used_rows = set()
        used_values = set()
        total_score = 0
        for value, row in cells:
            if row not in used_rows and value not in used_values:
                total_score += value
                used_rows.add(row)
                used_values.add(value)
        return total_score