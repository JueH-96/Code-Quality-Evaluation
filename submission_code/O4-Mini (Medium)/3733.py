from typing import List

class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        # Directions: UL, UR, DR, DL (clockwise)
        dirs = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
        # len_sfx[d][p][r][c]: from (r,c), moving in dirs[d], maximum t matching seq starting parity p
        # We only need two parities: p=0 -> expect 0, p=1 -> expect 2
        len_sfx = [
            [ [ [0]*m for _ in range(n) ] for _ in range(2) ]
            for _ in range(4)
        ]
        # Precompute suffix lengths
        for d in range(4):
            dr, dc = dirs[d]
            # Determine row iteration order
            if dr == 1:
                rows = range(n-1, -1, -1)
            else:
                rows = range(0, n)
            # Determine col iteration order
            if dc == 1:
                cols = range(m-1, -1, -1)
            else:
                cols = range(0, m)
            for p in (0, 1):
                # required value at parity p
                req = 2 if p == 1 else 0
                arr = len_sfx[d][p]
                arr_next = len_sfx[d][p^1]
                for r in rows:
                    for c in cols:
                        if grid[r][c] == req:
                            nr = r + dr
                            nc = c + dc
                            if 0 <= nr < n and 0 <= nc < m:
                                arr[r][c] = 1 + arr_next[nr][nc]
                            else:
                                arr[r][c] = 1
                        else:
                            arr[r][c] = 0
        # Compute prefix_len: starting at (r,c), dir d, sequence from index 0: [1,2,0,2,0,...]
        # prefix_len[r][c][d] = 0 if grid[r][c]!=1 else 1 + len_sfx from next cell with parity=1
        prefix_len = [
            [ [0]*m for _ in range(n) ]
            for _ in range(4)
        ]
        max_len = 0
        # Also track global max suffix length over all d,p
        global_max_sfx = 0
        for d in range(4):
            for p in (0,1):
                arr = len_sfx[d][p]
                for r in range(n):
                    # use Python's max to find max in row
                    row_max = max(arr[r])
                    if row_max > global_max_sfx:
                        global_max_sfx = row_max
        for d in range(4):
            dr, dc = dirs[d]
            pl = prefix_len[d]
            sfx = len_sfx[d][1]  # parity 1 suffix for index=1
            for r in range(n):
                for c in range(m):
                    if grid[r][c] == 1:
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < n and 0 <= nc < m:
                            pl[r][c] = 1 + sfx[nr][nc]
                        else:
                            pl[r][c] = 1
                        if pl[r][c] > max_len:
                            max_len = pl[r][c]
                    else:
                        pl[r][c] = 0
        # Now consider one-turn segments
        ans = max_len
        for r in range(n):
            for c in range(m):
                if grid[r][c] != 1:
                    continue
                # four possible start directions
                for d in range(4):
                    Lmax = prefix_len[d][r][c]
                    # prune if even with max suffix can't beat ans
                    if Lmax + global_max_sfx <= ans:
                        continue
                    dr, dc = dirs[d]
                    # outgoing direction
                    d2 = (d + 1) & 3
                    dr2, dc2 = dirs[d2]
                    # for each possible prefix length k+1
                    # we walk along the prefix
                    rr, cc = r, c
                    # L = k+1
                    for k in range(Lmax):
                        L = k + 1
                        # next parity for suffix
                        p = L & 1
                        # turning point is (rr,cc)
                        # next cell in new dir
                        nr2 = rr + dr2
                        nc2 = cc + dc2
                        rem = 0
                        if 0 <= nr2 < n and 0 <= nc2 < m:
                            rem = len_sfx[d2][p][nr2][nc2]
                        total = L + rem
                        if total > ans:
                            ans = total
                        # step rr,cc along original dir for next k
                        rr += dr
                        cc += dc
        return ans