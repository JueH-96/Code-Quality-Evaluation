class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        
        n = len(grid)
        m = len(grid[0])
        rowCount = [0] * n
        colCount = [0] * m
        
        # Precompute count of ones in each row and column.
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    rowCount[i] += 1
                    colCount[j] += 1
        
        result = 0
        # For every 1, it can be a pivot if there is at least one other 1 in its row and one in its column.
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    # minus one to exclude the pivot itself
                    extraInRow = rowCount[i] - 1
                    extraInCol = colCount[j] - 1
                    result += extraInRow * extraInCol
        
        return result