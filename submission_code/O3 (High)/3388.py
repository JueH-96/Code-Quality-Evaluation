from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        # Number of rows and columns
        n = len(grid)
        m = len(grid[0])
        
        # Pre-compute how many 1's are present in each row and in each column
        row_cnt = [0] * n
        col_cnt = [0] * m
        
        for i in range(n):
            rc = 0
            for j in range(m):
                if grid[i][j]:
                    rc += 1
                    col_cnt[j] += 1
            row_cnt[i] = rc
        
        # For every cell that contains 1, the number of right triangles
        # with the right angle at that cell equals
        #   (number of other 1's in its row)  *  (number of other 1's in its column)
        ans = 0
        for i in range(n):
            if row_cnt[i] < 2:      # no other 1 in this row – skip entire row
                continue
            for j in range(m):
                if grid[i][j] and col_cnt[j] > 1:
                    ans += (row_cnt[i] - 1) * (col_cnt[j] - 1)
        
        return ans