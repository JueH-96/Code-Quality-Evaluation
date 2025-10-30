from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        max_score = 0
        selected = [False] * rows
        
        for col in range(cols):
            current_score = 0
            unique_values = set()
            for row in range(rows):
                if not selected[row] and grid[row][col] not in unique_values:
                    current_score += grid[row][col]
                    unique_values.add(grid[row][col])
                    selected[row] = True
            max_score = max(max_score, current_score)
        
        return max_score