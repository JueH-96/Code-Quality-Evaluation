from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        prefix_sum = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                prefix_sum[i][j] = prefix_sum[i-1][j] + prefix_sum[i][j-1] - prefix_sum[i-1][j-1]
                if grid[i-1][j-1] == 'X':
                    prefix_sum[i][j] += 1
                elif grid[i-1][j-1] == 'Y':
                    prefix_sum[i][j] -= 1
        
        count = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                for x in range(i):
                    for y in range(j):
                        submatrix_sum = prefix_sum[i][j] - prefix_sum[x][j] - prefix_sum[i][y] + prefix_sum[x][y]
                        if submatrix_sum == 0 and grid[0][0] == 'X':
                            count += 1
        return count