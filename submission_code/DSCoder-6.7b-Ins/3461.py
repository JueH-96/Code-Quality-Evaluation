class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        left = [[0]*n for _ in range(m)]
        up = [[0]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    left[i][j] = j if j == 0 else left[i][j-1]
                    up[i][j] = i if i == 0 else up[i-1][j]

        right = [[n]*n for _ in range(m)]
        down = [[m]*n for _ in range(m)]
        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                if grid[i][j] == 1:
                    right[i][j] = j if j == n-1 else right[i][j+1]
                    down[i][j] = i if i == m-1 else down[i+1][j]

        res = float('inf')
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    for k in range(i+1, down[i][j]+1):
                        for l in range(j+1, right[i][j]+1):
                            if left[k][l] <= j and up[k][l] <= i:
                                res = min(res, (k-i+1)*(l-j+1))
        return res