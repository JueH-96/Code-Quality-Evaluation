from typing import List

class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if n == 0:
            return 0
        m = len(grid[0])
        N = n * m

        # Flatten grid for faster indexing
        grid_flat = [0] * N
        for i in range(n):
            row = grid[i]
            off = i * m
            for j in range(m):
                grid_flat[off + j] = row[j]

        # Four diagonal directions: down-right, down-left, up-left, up-right
        dr = [1,  1, -1, -1]
        dc = [1, -1, -1,  1]

        # dp_par_even_flat[d][pos] = max length of alternating sequence starting
        # at pos (included) along direction d with parity=0 at pos (expects val=0)
        # dp_par_odd_flat[d][pos]  = same with parity=1 at pos (expects val=2)
        dp_par_even_flat = [ [0] * N for _ in range(4) ]
        dp_par_odd_flat  = [ [0] * N for _ in range(4) ]

        # Build dp_par flats by dynamic programming along each direction
        for d in range(4):
            ddr, ddc = dr[d], dc[d]
            dp_e = dp_par_even_flat[d]
            dp_o = dp_par_odd_flat[d]

            # We need to process so that when we visit (i,j), its neighbor
            # (i+ddr, j+ddc) is already computed.
            if ddr > 0:
                i_iter = range(n-1, -1, -1)
            else:
                i_iter = range(n)
            if ddc > 0:
                j_iter = range(m-1, -1, -1)
            else:
                j_iter = range(m)

            for i in i_iter:
                base = i * m
                for j in j_iter:
                    pos = base + j
                    val = grid_flat[pos]
                    ni = i + ddr
                    nj = j + ddc
                    # check neighbor in bounds
                    if 0 <= ni < n and 0 <= nj < m:
                        npos = ni * m + nj
                        # parity-even (expects 0 here)
                        if val == 0:
                            dp_e[pos] = 1 + dp_o[npos]
                        else:
                            dp_e[pos] = 0
                        # parity-odd (expects 2 here)
                        if val == 2:
                            dp_o[pos] = 1 + dp_e[npos]
                        else:
                            dp_o[pos] = 0
                    else:
                        # neighbor out-of-bounds: only length=1 if matches
                        if val == 0:
                            dp_e[pos] = 1
                        else:
                            dp_e[pos] = 0
                        if val == 2:
                            dp_o[pos] = 1
                        else:
                            dp_o[pos] = 0

        # Precompute logs for sparse-table queries (max segment length ≤ max(n,m))
        max_len = max(n, m) + 1
        logs = [0] * (max_len + 1)
        for i in range(2, max_len + 1):
            logs[i] = logs[i // 2] + 1

        ans = 0

        # For each of the 4 directions, compute dp_start and handle V-shapes
        for d in range(4):
            ddr, ddc = dr[d], dc[d]
            dp_odd_d = dp_par_odd_flat[d]

            # dpS[pos] = length of pure prefix segment starting at pos in direction d
            # that starts with grid==1 then alternates (2,0,2,0,...)
            dpS = [0] * N
            # Compute dpS by scanning each cell
            for i in range(n):
                base = i * m
                for j in range(m):
                    pos = base + j
                    if grid_flat[pos] == 1:
                        ni = i + ddr
                        nj = j + ddc
                        if 0 <= ni < n and 0 <= nj < m:
                            dpS[pos] = 1 + dp_odd_d[ni * m + nj]
                        else:
                            dpS[pos] = 1
                        if dpS[pos] > ans:
                            ans = dpS[pos]
                    else:
                        dpS[pos] = 0

            # Now handle one-turn segments: first arm along d, then CW turn to new_d
            new_d = (d + 1) % 4
            dp_even_nd = dp_par_even_flat[new_d]
            dp_odd_nd  = dp_par_odd_flat[new_d]

            # Generate line-starts for direction d: cells whose backward neighbor is OOB
            if d == 0:    # down-right
                starts = [(0, j) for j in range(m)] + [(i, 0) for i in range(1, n)]
            elif d == 1:  # down-left
                starts = [(0, j) for j in range(m)] + [(i, m - 1) for i in range(1, n)]
            elif d == 2:  # up-left
                starts = [(n - 1, j) for j in range(m)] + [(i, m - 1) for i in range(0, n - 1)]
            else:         # up-right
                starts = [(n - 1, j) for j in range(m)] + [(i, 0) for i in range(0, n - 1)]

            # Process each diagonal line
            for (i0, j0) in starts:
                # collect positions along the line
                pos_list = []
                i, j = i0, j0
                while 0 <= i < n and 0 <= j < m:
                    pos_list.append(i * m + j)
                    i += ddr
                    j += ddc

                L_line = len(pos_list)
                if L_line < 2:
                    continue

                # Build arrA, arrB and dpL arrays of length L_line
                # arrA[t] = t + dp_par[new_d][pos_list[t]][par = t%2]
                # arrB[t] = t + dp_par[new_d][pos_list[t]][par = 1-t%2]
                arrA = [0] * L_line
                arrB = [0] * L_line
                dpL  = [0] * L_line
                for t in range(L_line):
                    p = pos_list[t]
                    dpL_t = dpS[p]
                    dpL[t] = dpL_t
                    ev = dp_even_nd[p]
                    od = dp_odd_nd[p]
                    if (t & 1) == 0:
                        # even t: arrA uses even-parity suffix, arrB uses odd-parity suffix
                        arrA[t] = t + ev
                        arrB[t] = t + od
                    else:
                        arrA[t] = t + od
                        arrB[t] = t + ev

                # Build sparse tables for arrA and arrB
                k_max = logs[L_line]
                stA = [arrA]
                stB = [arrB]
                for k in range(1, k_max + 1):
                    prevA = stA[k - 1]
                    prevB = stB[k - 1]
                    span = 1 << k
                    half = span >> 1
                    limit = L_line - span + 1
                    nxtA = [0] * limit
                    nxtB = [0] * limit
                    for idx in range(limit):
                        a1 = prevA[idx]
                        a2 = prevA[idx + half]
                        nxtA[idx] = a1 if a1 >= a2 else a2
                        b1 = prevB[idx]
                        b2 = prevB[idx + half]
                        nxtB[idx] = b1 if b1 >= b2 else b2
                    stA.append(nxtA)
                    stB.append(nxtB)

                # Query windows for each start index s
                for s in range(L_line):
                    span_len = dpL[s]
                    if span_len >= 2:
                        l = s + 1
                        r = s + span_len - 1
                        if r >= L_line:
                            r = L_line - 1
                        if l <= r:
                            length = r - l + 1
                            k = logs[length]
                            if (s & 1) == 0:
                                # use sparse table for arrA
                                row = stA[k]
                            else:
                                row = stB[k]
                            o = r - (1 << k) + 1
                            m1 = row[l]
                            m2 = row[o]
                            maxi = m1 if m1 >= m2 else m2
                            val = maxi - s
                            if val > ans:
                                ans = val

        return ans