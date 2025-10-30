class Solution:
    def minFlips(self, grid: List[List[int]]) -> int:
        def calculate_flips(arr):
            flips = 0
            n = len(arr)
            for i in range(n):
                j = n - 1 - i
                if i >= j:
                    break
                a = arr[i]
                b = arr[j]
                option0 = (a != 0) + (b != 0)
                option1 = (a != 1) + (b != 1)
                flips += min(option0, option1)
            return flips
        
        row_total = 0
        for row in grid:
            row_total += calculate_flips(row)
        
        col_total = 0
        m = len(grid)
        if m == 0:
            return 0
        n = len(grid[0])
        for c in range(n):
            column = [grid[r][c] for r in range(m)]
            col_total += calculate_flips(column)
        
        return min(row_total, col_total)