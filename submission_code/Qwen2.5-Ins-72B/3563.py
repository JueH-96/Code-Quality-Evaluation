from typing import List
from itertools import permutations

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        max_score = 0
        
        # Generate all possible column permutations
        for cols_perm in permutations(range(cols)):
            selected_values = set()
            score = 0
            for row in range(rows):
                col = cols_perm[row]
                value = grid[row][col]
                if value not in selected_values:
                    selected_values.add(value)
                    score += value
            max_score = max(max_score, score)
        
        return max_score