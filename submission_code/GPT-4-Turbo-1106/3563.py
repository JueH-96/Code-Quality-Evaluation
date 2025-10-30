from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        # Helper function to calculate the maximum score using backtracking
        def backtrack(row, visited, current_score):
            nonlocal max_score
            if row == len(grid):
                max_score = max(max_score, current_score)
                return
            for col in range(len(grid[row])):
                if grid[row][col] not in visited:
                    visited.add(grid[row][col])
                    backtrack(row + 1, visited, current_score + grid[row][col])
                    visited.remove(grid[row][col])
        
        max_score = 0
        backtrack(0, set(), 0)
        return max_score

# Example usage:
# sol = Solution()
# print(sol.maxScore([[1,2,3],[4,3,2],[1,1,1]]))  # Output: 8
# print(sol.maxScore([[8,7,6],[8,3,2]]))  # Output: 15