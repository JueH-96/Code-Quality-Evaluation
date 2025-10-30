from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        # Create a dictionary to store the maximum value for each column
        max_in_col = {}

        for j in range(n):
            max_val = 0
            for i in range(m):
                if grid[i][j] > max_val:
                    max_val = grid[i][j]
            max_in_col[j] = max_val

        # Sort the columns based on their maximum values in descending order
        sorted_cols = sorted(max_in_col.keys(), key=lambda x: -max_in_col[x])

        selected_values = set()
        score = 0

        for j in sorted_cols:
            for i in range(m):
                if grid[i][j] not in selected_values:
                    score += grid[i][j]
                    selected_values.add(grid[i][j])
                    break

        return score