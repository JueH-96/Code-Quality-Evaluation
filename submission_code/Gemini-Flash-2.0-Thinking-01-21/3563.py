from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols_in_row = len(grid[0]) if rows > 0 else 0
        memo = {}

        def solve(row_index, selected_values_tuple):
            if row_index == rows:
                return sum(selected_values_tuple)
            if (row_index, selected_values_tuple) in memo:
                return memo[(row_index, selected_values_tuple)]

            max_score = 0
            # Skip current row
            max_score = max(max_score, solve(row_index + 1, selected_values_tuple))

            # Select a cell in the current row
            for col_index in range(len(grid[row_index])):
                value = grid[row_index][col_index]
                if value not in selected_values_tuple:
                    new_selected_values_tuple = tuple(sorted(list(selected_values_tuple) + [value]))
                    max_score = max(max_score, value + solve(row_index + 1, new_selected_values_tuple))
            
            memo[(row_index, selected_values_tuple)] = max_score
            return max_score

        return solve(0, tuple())