class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        dirs = ((1, 1), (1, -1), (-1, -1), (-1, 1))          # 0,1,2,3
        pre  = [[[0]*4 for _ in range(m)] for _ in range(n)]  # prefix – ends here
        alt2 = [[[0]*4 for _ in range(m)] for _ in range(n)]  # suffix starting with 2
        alt0 = [[[0]*4 for _ in range(m)] for _ in range(n)]  # suffix starting with 0
        
        # ---------- build pre ----------
        for d,(dr,dc) in enumerate(dirs):
            rows = range(n) if dr==1 else range(n-1,-1,-1)
            for r in rows:
                cols = range(m) if dc==1 else range(m-1,-1,-1)
                for c in cols:
                    val = grid[r][c]
                    if val==1:                         # new start
                        pre[r][c][d] = 1
                    else:
                        pr, pc = r-dr, c-dc
                        if 0<=pr<n and 0<=pc<m and pre[pr][pc][d]:
                            exp = 2 if pre[pr][pc][d] & 1 else 0
                            if val == exp:
                                pre[r][c][d] = pre[pr][pc][d] + 1
        # ---------- build alternating suffixes ----------
        for d,(dr,dc) in enumerate(dirs):
            rows = range(n-1,-1,-1) if dr==1 else range(n)   # reversed direction
            for r in rows:
                cols = range(m-1,-1,-1) if dc==1 else range(m)
                for c in cols:
                    nr, nc = r+dr, c+dc
                    nxt2 = nxt0 = 0
                    if 0<=nr<n and 0<=nc<m:
                        nxt2 = alt2[nr][nc][d]
                        nxt0 = alt0[nr][nc][d]
                    val = grid[r][c]
                    if val == 2:
                        alt2[r][c][d] = 1 + nxt0            # next must be 0
                    if val == 0:
                        alt0[r][c][d] = 1 + nxt2            # next must be 2
        # ---------- evaluate ----------
        best = 0
        for r in range(n):
            for c in range(m):
                for d in range(4):
                    L1 = pre[r][c][d]
                    best = max(best, L1)                    # straight diagonal
                    if L1:
                        need2 = (L1 & 1)                    # odd length → need 2
                        dr2, dc2 = dirs[(d+1)&3]            # clockwise direction
                        nr, nc = r+dr2, c+dc2
                        if 0<=nr<n and 0<=nc<m:
                            if need2:
                                L2 = alt2[nr][nc][(d+1)&3]
                            else:
                                L2 = alt0[nr][nc][(d+1)&3]
                            best = max(best, L1 + L2)
        return best