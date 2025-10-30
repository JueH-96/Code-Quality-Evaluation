from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        prefix_sum = [[0] * n for _ in range(m)]
        count = 0
        
        for i in range(m):
            for j in range(n):
                prefix_sum[i][j] = (1 if grid[i][j] == 'X' else 0) + (prefix_sum[i-1][j] if i > 0 else 0) + (prefix_sum[i][j-1] if j > 0 else 0) - (prefix_sum[i-1][j-1] if i > 0 and j > 0 else 0)
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 'X':
                    x_count = 1 if grid[i][j] == 'X' else 0
                    y_count = 0
                    for k in range(i, -1, -1):
                        for l in range(j, -1, -1):
                            if k == i and l == j:
                                continue
                            if grid[k][l] == 'Y':
                                y_count += 1
                            if x_count == y_count and x_count > 0:
                                count += 1
                            if x_count > y_count or (k == 0 and l == 0):
                                break
        
        return count