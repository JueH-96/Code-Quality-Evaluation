class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        
        # Calculate minimum flips needed to make all rows palindromic
        row_flips = 0
        for i in range(rows):
            for j in range(cols // 2):
                # Cost to make both positions 0
                cost_to_0 = (grid[i][j] == 1) + (grid[i][cols - j - 1] == 1)
                # Cost to make both positions 1
                cost_to_1 = (grid[i][j] == 0) + (grid[i][cols - j - 1] == 0)
                # Choose the lower cost
                row_flips += min(cost_to_0, cost_to_1)
        
        # Calculate minimum flips needed to make all columns palindromic
        col_flips = 0
        for j in range(cols):
            for i in range(rows // 2):
                # Cost to make both positions 0
                cost_to_0 = (grid[i][j] == 1) + (grid[rows - i - 1][j] == 1)
                # Cost to make both positions 1
                cost_to_1 = (grid[i][j] == 0) + (grid[rows - i - 1][j] == 0)
                # Choose the lower cost
                col_flips += min(cost_to_0, cost_to_1)
        
        return min(row_flips, col_flips)