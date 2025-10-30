from typing import List

class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        # Compute the number of 1's in each row and each column
        m, n = len(grid), len(grid[0])
        row_counts = [0] * m
        col_counts = [0] * n
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    row_counts[i] += 1
                    col_counts[j] += 1
        
        # For each 1-cell as the right-angle vertex, the number of right triangles
        # with that vertex is (ones in same row minus itself) * (ones in same column minus itself).
        total = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    total += (row_counts[i] - 1) * (col_counts[j] - 1)
        
        return total