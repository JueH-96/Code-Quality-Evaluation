from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        # Count of 1's in each row and each column
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        
        row_count = [sum(row) for row in grid]
        col_count = [0] * m
        for j in range(m):
            cnt = 0
            for i in range(n):
                cnt += grid[i][j]
            col_count[j] = cnt
        
        # For each cell that is 1, it can serve as the right-angle vertex.
        # We pick any other 1 in the same row and any other 1 in the same column.
        ans = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    # choices in row i excluding (i,j): row_count[i] - 1
                    # choices in col j excluding (i,j): col_count[j] - 1
                    ans += (row_count[i] - 1) * (col_count[j] - 1)
        
        return ans