class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        n = len(grid)
        
        def solve(row, selected_values):
            if row == n:
                return 0
            
            max_score = solve(row + 1, selected_values) 
            
            for col in range(len(grid[row])):
                value = grid[row][col]
                if value not in selected_values:
                    new_selected_values = selected_values.copy()
                    new_selected_values.add(value)
                    max_score = max(max_score, value + solve(row + 1, new_selected_values))
            
            return max_score
        
        return solve(0, set())