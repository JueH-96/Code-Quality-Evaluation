class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        max_score = 0

        def backtrack(row, current_score, selected_values):
            nonlocal max_score
            if row == rows:
                max_score = max(max_score, current_score)
                return

            backtrack(row + 1, current_score, selected_values)

            for col in range(cols):
                value = grid[row][col]
                if value not in selected_values:
                    new_selected_values = set(selected_values)
                    new_selected_values.add(value)
                    backtrack(row + 1, current_score + value, new_selected_values)

        backtrack(0, 0, set())
        return max_score