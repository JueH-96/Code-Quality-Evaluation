class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        num_dirs = len(dirs)
        
        run20 = [[[0] * m for _ in range(n)] for _ in range(num_dirs)]
        run02 = [[[0] * m for _ in range(n)] for _ in range(num_dirs)]
        
        for d in range(num_dirs):
            for i in range(n):
                for j in range(m):
                    if grid[i][j] == 2:
                        run20[d][i][j] = 1
                    elif grid[i][j] == 0:
                        run02[d][i][j] = 1
        
        for d in range(num_dirs):
            dx, dy = dirs[d]
            if d == 0:
                for i in range(n-1, -1, -1):
                    for j in range(m-1, -1, -1):
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < n and 0 <= nj < m:
                            if grid[i][j] == 2:
                                run20[d][i][j] = 1 + run02[d][ni][nj]
                            if grid[i][j] == 0:
                                run02[d][i][j] = 1 + run20[d][ni][nj]
            elif d == 1:
                for i in range(n-1, -1, -1):
                    for j in range(0, m):
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < n and 0 <= nj < m:
                            if grid[i][j] == 2:
                                run20[d][i][j] = 1 + run02[d][ni][nj]
                            if grid[i][j] == 0:
                                run02[d][i][j] = 1 + run20[d][ni][nj]
            elif d == 2:
                for i in range(0, n):
                    for j in range(m-1, -1, -1):
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < n and 0 <= nj < m:
                            if grid[i][j] == 2:
                                run20[d][i][j] = 1 + run02[d][ni][nj]
                            if grid[i][j] == 0:
                                run02[d][i][j] = 1 + run20[d][ni][nj]
            elif d == 3:
                for i in range(0, n):
                    for j in range(0, m):
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < n and 0 <= nj < m:
                            if grid[i][j] == 2:
                                run20[d][i][j] = 1 + run02[d][ni][nj]
                            if grid[i][j] == 0:
                                run02[d][i][j] = 1 + run20[d][ni][nj]
        
        ans = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] != 1:
                    continue
                for d in range(num_dirs):
                    dxd, dyd = dirs[d]
                    length = 1
                    ni, nj = i + dxd, j + dyd
                    if 0 <= ni < n and 0 <= nj < m:
                        length += run20[d][ni][nj]
                    for k in range(length):
                        x = i + k * dxd
                        y = j + k * dyd
                        nd = (d + 1) % 4
                        dxn, dyn = dirs[nd]
                        nx, ny = x + dxn, y + dyn
                        ext = 0
                        if 0 <= nx < n and 0 <= ny < m:
                            if k % 2 == 0:
                                ext = run20[nd][nx][ny]
                            else:
                                ext = run02[nd][nx][ny]
                        total = k + 1 + ext
                        if total > ans:
                            ans = total
        return ans