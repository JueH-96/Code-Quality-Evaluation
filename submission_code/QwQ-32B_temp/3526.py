class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        m = len(grid)
        if m == 0:
            return 0
        n = len(grid[0])
        
        row_cost = 0
        for row in grid:
            cost = 0
            for i in range(n // 2):
                if row[i] != row[~i]:
                    cost += 1
            row_cost += cost
        
        column_cost = 0
        for col in range(n):
            cost = 0
            for i in range(m // 2):
                a = grid[i][col]
                b = grid[m - 1 - i][col]
                if a != b:
                    cost += 1
            column_cost += cost
        
        return min(row_cost, column_cost)