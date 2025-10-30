class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        def backtrack(row, used_values, curr_sum):
            if row == len(grid):
                return curr_sum
            
            max_score = curr_sum
            
            # Try each cell in current row
            for col in range(len(grid[0])):
                val = grid[row][col]
                # Only use value if not used before
                if val not in used_values:
                    used_values.add(val)
                    score = backtrack(row + 1, used_values, curr_sum + val)
                    max_score = max(max_score, score)
                    used_values.remove(val)
            
            # Can also skip this row entirely
            score = backtrack(row + 1, used_values, curr_sum)
            max_score = max(max_score, score)
            
            return max_score
            
        return backtrack(0, set(), 0)