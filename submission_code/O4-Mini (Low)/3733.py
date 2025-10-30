from typing import List

class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        # Directions: 0: NE(-1,+1), 1: SE(+1,+1), 2: SW(+1,-1), 3: NW(-1,-1)
        dirs = [(-1,1),(1,1),(1,-1),(-1,-1)]
        # clockwise turn map: d -> (d+1)%4 in our ordering
        # parity 1 expects 2, parity 0 expects 0
        # dp_par[b][i][j][d]: suffix length starting at (i,j) expecting parity b
        dp_par = [
            [ [ [0]*4 for _ in range(m) ] for __ in range(n) ],
            [ [ [0]*4 for _ in range(m) ] for __ in range(n) ]
        ]
        # Compute dp_par by scanning in reverse of each direction
        for d, (dr,dc) in enumerate(dirs):
            # choose scanning order so that (i+dr,j+dc) is done before (i,j)
            rows = range(n-1, -1, -1) if dr>0 else range(n)
            cols = range(m-1, -1, -1) if dc>0 else range(m)
            for i in rows:
                for j in cols:
                    ni, nj = i+dr, j+dc
                    for b in (0,1):
                        exp = 2 if b==1 else 0
                        if grid[i][j] == exp:
                            nxt = 0
                            if 0 <= ni < n and 0 <= nj < m:
                                nxt = dp_par[1-b][ni][nj][d]
                            dp_par[b][i][j][d] = 1 + nxt
                        else:
                            dp_par[b][i][j][d] = 0

        # f[i][j][d]: max sequence length ending exactly at (i,j) without turn, along dir d
        f = [ [ [0]*4 for _ in range(m) ] for __ in range(n) ]
        ans = 0

        # Compute f by forward scan in each dir
        for d, (dr,dc) in enumerate(dirs):
            rows = range(n) if dr>0 else range(n-1, -1, -1)
            cols = range(m) if dc>0 else range(m-1, -1, -1)
            for i in rows:
                for j in cols:
                    # start new segment if it's a '1'
                    if grid[i][j] == 1:
                        f[i][j][d] = 1
                    # try extend from previous
                    pi, pj = i-dr, j-dc
                    if 0 <= pi < n and 0 <= pj < m and f[pi][pj][d] > 0:
                        prev_len = f[pi][pj][d]
                        idx_prev = prev_len - 1
                        # next index = idx_prev + 1
                        b = (idx_prev + 1) & 1
                        exp = 2 if b==1 else 0
                        if grid[i][j] == exp:
                            f[i][j][d] = max(f[i][j][d], prev_len + 1)
                    # update answer with no turn
                    ans = max(ans, f[i][j][d])

        # consider one clockwise turn at (i,j)
        for i in range(n):
            for j in range(m):
                for d in range(4):
                    pre = f[i][j][d]
                    if pre == 0:
                        continue
                    idx_here = pre - 1
                    # next parity after this cell
                    bnext = (idx_here + 1) & 1
                    d2 = (d + 1) & 3
                    ni, nj = i + dirs[d2][0], j + dirs[d2][1]
                    suf = 0
                    if 0 <= ni < n and 0 <= nj < m:
                        suf = dp_par[bnext][ni][nj][d2]
                    ans = max(ans, pre + suf)

        return ans