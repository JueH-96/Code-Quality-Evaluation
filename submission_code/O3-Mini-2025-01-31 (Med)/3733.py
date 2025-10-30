class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        
        # Helper functions.
        # expected(s) returns the expected value given state s.
        # States:
        #   0: expecting 1 (only allowed at the very beginning)
        #   1: expecting 2
        #   2: expecting 0
        def expected(s: int) -> int:
            if s == 0:
                return 1
            elif s == 1:
                return 2
            else:
                return 0
        
        # state transition: next_state(s)
        def next_state(s: int) -> int:
            if s == 0:
                return 1
            elif s == 1:
                return 2
            else:
                # from state 2, next expected becomes 2 again? No – note the pattern:
                # Pattern: [1, 2, 0, 2, 0, 2, 0, ...]
                # So after state 2, the next expected is always 2? Actually, see:
                # index: 0 -> 1, 1 -> 2, 2 -> 0, 3 -> 2, 4 -> 0, ...
                # So the transition after 2 (which came at index 2 or 4 or ...) is 1? Let's re-read:
                # After the first element (1), the sequence is: 2, 0, 2, 0, ... 
                # So if state 0 means "need 1", then once you see that, next is state 1 meaning "need 2"
                # then after 2, next should be state 2 (need 0), and after that, next should be state 1 (need 2), and so on.
                return 1
        
        # clockwise turn utility: for a direction (dr, dc), a clockwise 90-degree rotation 
        # sends (dr, dc) -> (dc, -dr)
        def clockwise(turn):
            dr, dc = turn
            return (dc, -dr)
        
        # Our possible diagonal directions:
        dirs = [(1,1), (1,-1), (-1,1), (-1,-1)]
        D = len(dirs)
        
        # We'll build two DP tables for each direction.
        # forward_dp[d][r][c][s]: maximum chain length starting at (r,c) in direction d, 
        #   if we expect f(s) at (r,c), then next state is next_state(s).
        # We need 3 states (s = 0, 1, 2).
        forward_dp = [ [ [ [0]*3 for _ in range(m) ] for _ in range(n) ] for _ in range(D) ]
        
        # rev_dp[d][r][c]: maximum chain length ending at (r,c) along direction d,
        #  following the pattern starting at 1. (Here there's no "state" dimension because
        #  if a cell is reached as part of a chain it must be the unique possibility from that chain.)
        rev_dp = [ [ [0]*m for _ in range(n) ] for _ in range(D) ]
        
        # Compute forward_dp for each direction d.
        for d, (dr, dc) in enumerate(dirs):
            # determine iteration order to "ensure" that the next cell in direction is computed already.
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
                    for s in range(3):
                        # if the current cell matches expected character for state s:
                        if grid[r][c] == expected(s):
                            best = 1
                            nr = r + dr
                            nc = c + dc
                            # if next cell in-bound, we can extend the chain:
                            if 0 <= nr < n and 0 <= nc < m:
                                ns = next_state(s)
                                nxt = forward_dp[d][nr][nc][ns]
                                if nxt:  # if extension is available
                                    best = 1 + nxt
                            forward_dp[d][r][c][s] = best
                        else:
                            forward_dp[d][r][c][s] = 0
                            
        # Compute rev_dp for each direction d.
        # In this DP we want the maximum chain length ending at (r,c)
        # along the given diagonal direction, with the segment starting with 1.
        for d, (dr, dc) in enumerate(dirs):
            # In rev_dp, the transition is from the previous cell in the chain:
            #  pre = (r - dr, c - dc)
            # And cell (r, c) can start a chain (of length 1) if grid[r][c] == 1.
            if dr > 0:
                # Then (r-dr) < r, so we iterate r increasing.
                r_range = range(0, n)
            else:
                r_range = range(n-1, -1, -1)
            if dc > 0:
                c_range = range(0, m)
            else:
                c_range = range(m-1, -1, -1)
                
            for r in r_range:
                for c in c_range:
                    candidate = 0
                    # always a possibility to start a new segment at (r,c) if cell equals 1:
                    if grid[r][c] == 1:
                        candidate = 1
                    pr = r - dr
                    pc = c - dc
                    if 0 <= pr < n and 0 <= pc < m and rev_dp[d][pr][pc] > 0:
                        # Let L = rev_dp[d][pr][pc] be the chain length ending at predecessor.
                        # Then for (r, c), the expected value is given by expected(L).
                        if grid[r][c] == expected(rev_dp[d][pr][pc]):
                            candidate = max(candidate, rev_dp[d][pr][pc] + 1)
                    rev_dp[d][r][c] = candidate
        
        ans = 0
        # First consider "straight" segments (i.e. without any turn).
        for d in range(D):
            for r in range(n):
                for c in range(m):
                    # The segment must begin with state 0 (i.e., expecting 1).
                    if forward_dp[d][r][c][0] > ans:
                        ans = forward_dp[d][r][c][0]
        
        # Next consider V-shaped segments (with one clockwise turn).
        # For each direction d for arm1, let d2 be the clockwise direction.
        # The idea: we want to “glue” an arm1 ending at vertex (r,c) coming from direction d 
        # with an arm2 starting at (r,c) in direction d2.
        # However, note that the expected state for arm2 at vertex depends on the arm1 length L.
        # Recall that overall pattern: index0 = 1, index1 = 2, index2 = 0, index3 = 2, index4 = 0, ...
        # So if arm1 length L is chosen then the vertex is at index L-1.
        # Let req_state be:
        #   if L == 1, then req_state = 0 (expecting 1, since pattern[0] = 1)
        #   if L >= 2, then if (L-1) is odd then pattern[L-1] = 2 (state 1) else if even then pattern[L-1] = 0 (state 2).
        for d, (dr, dc) in enumerate(dirs):
            d2 = None
            # compute clockwise of direction d:
            d2_vec = clockwise((dr, dc))
            # find index of d2_vec in dirs
            for i, vect in enumerate(dirs):
                if vect == d2_vec:
                    d2 = i
                    break
            if d2 is None:
                continue
            for r in range(n):
                for c in range(m):
                    # Here, (r,c) acts as the vertex.
                    # It could be the endpoint of an arm1 coming in direction d.
                    # There are two possibilities:
                    #   (a) Arm1 is just the vertex (starting and ending at (r,c)) if grid[r][c]==1.
                    #   (b) Arm1 is an extension from an earlier cell, i.e. rev_dp[d][r][c] might be > 1.
                    if rev_dp[d][r][c] == 0:
                        continue
                    # Possibility 1: length 1 (always possible if grid[r][c]==1).
                    for L in (1, rev_dp[d][r][c] if rev_dp[d][r][c] > 1 else 1):
                        # If the rev_dp is >1 then that is the only possibility from continuation.
                        # (But note that the possibility L==1 is always available (starting fresh at the vertex).
                        req_state = 0 if L == 1 else (1 if ((L-1) & 1) == 1 else 2)
                        # For arm2 (starting at vertex) along d2, using the computed forward_dp.
                        arm2 = forward_dp[d2][r][c][req_state]
                        if arm2 > 0:
                            total = L + arm2 - 1  # subtract 1 because the vertex is shared.
                            if total > ans:
                                ans = total
        return ans