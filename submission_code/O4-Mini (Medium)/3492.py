from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        
        # Build prefix sums for 'X' and 'Y'. We use (n+1)x(m+1) arrays
        # so that prefix[i][j] is sum over grid[0..i-1][0..j-1].
        px = [[0] * (m + 1) for _ in range(n + 1)]
        py = [[0] * (m + 1) for _ in range(n + 1)]
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                isX = 1 if grid[i-1][j-1] == 'X' else 0
                isY = 1 if grid[i-1][j-1] == 'Y' else 0
                # 2D prefix sum relation
                px[i][j] = px[i-1][j] + px[i][j-1] - px[i-1][j-1] + isX
                py[i][j] = py[i-1][j] + py[i][j-1] - py[i-1][j-1] + isY
        
        # Count all submatrices from (0,0) to (i,j) that
        # 1. include at least one 'X'
        # 2. have equal #X and #Y
        ans = 0
        for i in range(1, n+1):
            for j in range(1, m+1):
                cx = px[i][j]
                cy = py[i][j]
                if cx > 0 and cx == cy:
                    ans += 1
        
        return ans