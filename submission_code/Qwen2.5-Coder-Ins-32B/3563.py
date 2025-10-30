from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        def backtrack(row, used, current_sum):
            if row == len(grid):
                return current_sum
            
            max_score = 0
            for col in range(len(grid[row])):
                if grid[row][col] not in used:
                    used.add(grid[row][col])
                    max_score = max(max_score, backtrack(row + 1, used, current_sum + grid[row][col]))
                    used.remove(grid[row][col])
            
            return max_score
        
        return backtrack(0, set(), 0)