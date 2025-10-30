from typing import List

class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        # There are only three "states": 
        # state 0 means we expect a 1,
        # state 1 means we expect a 2,
        # state 2 means we expect a 0.
        # And the sequence 1 (state0) -> 2 (state1) -> 0 (state2) -> 2 (state1) -> 0 (state2) -> ...
        req = [1, 2, 0]
        # Next state transitions:
        # state0 -> state1; state1 -> state2; state2 -> state1.
        nxt = [1, 2, 1]
        
        # Our four diagonal directions.
        # d0: down-right, d1: down-left, d2: up-left, d3: up-right.
        directions = [(1,1), (1,-1), (-1,-1), (-1,1)]
        
        # We will precompute 4 global DP arrays.
        # For each direction d and each state s∈{0,1,2} we compute a matrix dp_d[s][r][c]
        # which equals the maximum length of a valid chain that starts at (r,c) in that direction,
        # assuming we require the number req[s] at (r,c).
        dp_global = []   # list of 4 items; each is a tuple (dp_state0, dp_state1, dp_state2)
        
        for dr, dc in directions:
            dp0 = [[0]*m for _ in range(n)]
            dp1 = [[0]*m for _ in range(n)]
            dp2 = [[0]*m for _ in range(n)]
            # We must choose the iteration order such that for a given cell (r,c) the neighbour (r+dr, c+dc)
            # is already computed.
            if dr > 0:
                r_range = range(n-1, -1, -1)
            else:
                r_range = range(0, n)
            if dc > 0:
                c_range = range(m-1, -1, -1)
            else:
                c_range = range(0, m)
            for r in r_range:
                for c in c_range:
                    # state0: expecting 1
                    if grid[r][c] == req[0]:
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < n and 0 <= nc < m:
                            dp0[r][c] = 1 + dp1[nr][nc]
                        else:
                            dp0[r][c] = 1
                    else:
                        dp0[r][c] = 0
                    # state1: expecting 2
                    if grid[r][c] == req[1]:
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < n and 0 <= nc < m:
                            dp1[r][c] = 1 + dp2[nr][nc]
                        else:
                            dp1[r][c] = 1
                    else:
                        dp1[r][c] = 0
                    # state2: expecting 0
                    if grid[r][c] == req[2]:
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < n and 0 <= nc < m:
                            dp2[r][c] = 1 + dp1[nr][nc]
                        else:
                            dp2[r][c] = 1
                    else:
                        dp2[r][c] = 0
            dp_global.append( (dp0, dp1, dp2) )
        
        # A helper function to generate all "diagonals" (lists of coordinates) for a given first-leg direction.
        # We “start” from every cell that cannot be reached from a previous cell in the reverse of d.
        def get_diagonals(d):
            dr, dc = d
            diag_list = []
            for r in range(n):
                for c in range(m):
                    # If the "previous" cell along this direction is out‐of‐bounds then (r,c) is a source.
                    pr = r - dr
                    pc = c - dc
                    if pr < 0 or pr >= n or pc < 0 or pc >= m:
                        # Follow the diagonal starting at (r,c)
                        cur = (r, c)
                        cur_diag = []
                        while 0 <= cur[0] < n and 0 <= cur[1] < m:
                            cur_diag.append(cur)
                            cur = (cur[0] + dr, cur[1] + dc)
                        diag_list.append(cur_diag)
            return diag_list
        
        ans = 0  # overall answer
        
        # Now we “try” every first-leg direction. (We will use the rev–dp along the diagonal.)
        # For a given first–leg direction d and turn direction d′ = clockwise(d)
        # (note: (dr,dc) rotated clockwise becomes (dc, -dr) )
        for d_index in range(4):
            d = directions[d_index]
            # Compute d′ = clockwise(d): (dr,dc) -> (dc, -dr)
            dprime = (d[1], -d[0])
            # dprime is the direction of the second leg.
            dprime_index = (d_index + 1) % 4  # Because our directions are arranged cyclically.
            
            # Get the diagonals in the first‐leg direction d.
            diags = get_diagonals(d)
            for diag in diags:
                Ld = len(diag)
                if Ld == 0: 
                    continue
                # For the first leg we “simulate” a contiguous segment along this diagonal.
                # (We do a “reverse‐DP” along the diagonal in the order of movement.)
                # Here the valid sequence must be: 1,2,0,2,0,...
                # That is, if we denote by rev[i] the valid segment length ending at diag[i],
                # then we define for i=0:
                #    rev[0] = 1 if grid[diag[0]] == 1 else 0.
                # And for i > 0, if rev[i-1] > 0 then the next expected number is:
                #    2 if rev[i-1] is odd  (since if we saw 1 then next must be 2)
                #    0 if rev[i-1] is even (if we already got [1,2] then next must be 0, etc.)
                # Otherwise, if diag[i] is a “1” then we start a new segment.
                rev = [0] * Ld
                r0, c0 = diag[0]
                rev[0] = 1 if grid[r0][c0] == 1 else 0
                # update answer with straight–segment candidate
                if rev[0] > ans:
                    ans = rev[0]
                for i in range(1, Ld):
                    r,c = diag[i]
                    prev = rev[i-1]
                    # If we could extend from cell i–1 then the expected number is:
                    if prev > 0:
                        exp = 2 if (prev & 1)==1 else 0
                    else:
                        exp = 1  # new segment must start with 1
                    if grid[r][c] == exp:
                        rev[i] = prev + 1
                    elif grid[r][c] == 1:
                        rev[i] = 1
                    else:
                        rev[i] = 0
                    if rev[i] > ans:
                        ans = rev[i]
                
                # Now, for every cell in this diagonal we consider the possibility of a turn at that cell.
                # At diag[i] the first–leg segment has length L = rev[i]. Then the second–leg must start
                # at the neighbor (r + dprime[0], c + dprime[1]) in the second–leg direction.
                # Moreover, the expected value (state) for the second leg is 2 if L is odd (because then we expect 2),
                # or 0 if L is even.
                for i in range(Ld):
                    L_first = rev[i]
                    if L_first == 0:
                        continue
                    r,c = diag[i]
                    nr = r + dprime[0]
                    nc = c + dprime[1]
                    if nr < 0 or nr >= n or nc < 0 or nc >= m:
                        sec = 0
                    else:
                        # if L_first is odd, next expected is 2 → state 1; if even, next expected is 0 → state 2.
                        st = 1 if (L_first & 1)==1 else 2
                        sec = dp_global[dprime_index][st][nr][nc]
                    cand = L_first + sec
                    if cand > ans:
                        ans = cand
        
        return ans