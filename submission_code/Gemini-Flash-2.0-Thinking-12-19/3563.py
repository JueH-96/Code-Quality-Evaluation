class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        max_score = 0

        def backtrack(row_index, selected_values, current_sum):
            nonlocal max_score
            if row_index == rows:
                max_score = max(max_score, current_sum)
                return

            # Option 1: Skip current row
            backtrack(row_index + 1, selected_values, current_sum)

            # Option 2: Select a cell from current row
            for col_index in range(cols):
                value = grid[row_index][col_index]
                if value not in selected_values:
                    new_selected_values = set(selected_values)
                    new_selected_values.add(value)
                    backtrack(row_index + 1, new_selected_values, current_sum + value)

        backtrack(0, set(), 0)
        return max_score