class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        count = 0
        
        # For each potential right angle vertex
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    # Count 1s in same row
                    row_ones = []
                    for col in range(n):
                        if col != j and grid[i][col] == 1:
                            row_ones.append(col)
                    
                    # Count 1s in same column
                    col_ones = []
                    for row in range(m):
                        if row != i and grid[row][j] == 1:
                            col_ones.append(row)
                    
                    # For each pair of row and column 1s, check if they form right triangle
                    for r in row_ones:
                        for c in col_ones:
                            if grid[c][r] == 1:
                                count += 1
        
        return count