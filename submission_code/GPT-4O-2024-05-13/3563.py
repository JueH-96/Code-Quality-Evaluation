from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        from itertools import permutations
        
        n = len(grid)
        m = len(grid[0])
        
        max_score = 0
        
        # Generate all permutations of column indices
        for perm in permutations(range(m), n):
            current_score = 0
            used_values = set()
            valid = True
            
            for i in range(n):
                value = grid[i][perm[i]]
                if value in used_values:
                    valid = False
                    break
                used_values.add(value)
                current_score += value
            
            if valid:
                max_score = max(max_score, current_score)
        
        return max_score