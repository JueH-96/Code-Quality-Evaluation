from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        from itertools import permutations
        
        max_score = 0
        rows = len(grid)
        cols = len(grid[0])
        
        # Generate all permutations of column indices
        for perm in permutations(range(cols), rows):
            selected_values = set()
            current_score = 0
            valid_selection = True
            
            for r in range(rows):
                value = grid[r][perm[r]]
                if value in selected_values:
                    valid_selection = False
                    break
                selected_values.add(value)
                current_score += value
            
            if valid_selection:
                max_score = max(max_score, current_score)
        
        return max_score