from typing import List
from itertools import permutations

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        max_score = 0
        
        # Generate all possible column permutations
        for perm in permutations(range(cols), rows):
            selected_values = set()
            score = 0
            for i in range(rows):
                if grid[i][perm[i]] not in selected_values:
                    selected_values.add(grid[i][perm[i]])
                    score += grid[i][perm[i]]
            max_score = max(max_score, score)
        
        return max_score