class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        import copy
        num_rows = len(grid)
        num_cols = len(grid[0]) if num_rows > 0 else 0
        sorted_grid = [sorted(row, reverse=True) for row in grid]
        
        best_sum = 0
        
        def backtrack(row, selected_values, current_sum):
            nonlocal best_sum
            if row == num_rows:
                best_sum = max(best_sum, current_sum)
                return
            for value in sorted_grid[row]:
                if value not in selected_values:
                    new_selected = selected_values.copy()
                    new_selected.add(value)
                    backtrack(row + 1, new_selected, current_sum + value)
            # Pruning: estimate the upper bound for the remaining sum
            remaining_sum = current_sum
            for r in range(row, num_rows):
                for val in sorted_grid[r]:
                    if val not in selected_values:
                        remaining_sum += val
                        break
            if remaining_sum <= best_sum:
                return
        
        # Start backtracking from the first row with an empty set
        backtrack(0, set(), 0)
        return best_sum