from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        memo = {}

        def solve(row, selected_values):
            if row == rows:
                return 0

            if (row, tuple(sorted(selected_values))) in memo:
                return memo[(row, tuple(sorted(selected_values)))]

            # Option 1: Don't select any cell from the current row
            max_score = solve(row + 1, selected_values)

            # Option 2: Select one cell from the current row
            for col in range(cols):
                value = grid[row][col]
                if value not in selected_values:
                    max_score = max(max_score, value + solve(row + 1, selected_values | {value}))

            memo[(row, tuple(sorted(selected_values)))] = max_score
            return max_score

        return solve(0, frozenset())