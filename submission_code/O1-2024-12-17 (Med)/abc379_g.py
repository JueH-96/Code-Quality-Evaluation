def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    H = int(input_data[0])
    W = int(input_data[1])
    grid_strings = input_data[2:]

    MOD = 998244353

    # ----------------------------------------------------------------
    # 1) Parse the input grid, store forced colors as integers 0..2, or -1 for '?'
    #    Also check immediately for any adjacent conflict in the given digits.
    # ----------------------------------------------------------------
    forced = [[-1]*W for _ in range(H)]  # forced[r][c] in {0,1,2} if forced, -1 otherwise
    # map '1','2','3' -> 0,1,2
    color_map = {'1':0, '2':1, '3':2}

    def neighbors(r, c):
        for nr, nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if 0 <= nr < H and 0 <= nc < W:
                yield nr, nc

    # Fill forced and check for direct conflicts
    idx = 0
    for r in range(H):
        row_str = grid_strings[r]
        for c in range(W):
            ch = row_str[c]
            if ch != '?':
                forced[r][c] = color_map[ch]
    
    # Check for conflict among forced neighbors
    for r in range(H):
        for c in range(W):
            if forced[r][c] != -1:
                for nr, nc in neighbors(r, c):
                    if forced[nr][nc] == forced[r][c] and forced[r][c] != -1:
                        # conflict
                        print(0)
                        return

    # ----------------------------------------------------------------
    # If H=1 or W=1, the grid is just a line. We can solve in O(H*W) easily:
    # We go along the line (row or column), each cell must differ from the previous cell.
    #
    # Count how many ways to assign '?' so that no two adjacent are equal.
    # ----------------------------------------------------------------
    if H == 1 and W == 1:
        # single cell
        if forced[0][0] == -1:
            print(3 % MOD)  # can be 1,2,3
        else:
            print(1 % MOD)  # forced color
        return

    if H == 1 or W == 1:
        # It is effectively a 1D chain of length n = H*W
        # Let n = H*W. We read them in order and ensure adjacent differ.
        n = H * W
        line = []
        if H == 1:
            for c in range(W):
                line.append(forced[0][c])
        else:  # W == 1
            for r in range(H):
                line.append(forced[r][0])
        
        # dp[i][color] = number of ways to color up to index i with cell i = color
        # color in {0,1,2}.
        # If line[i] != -1, that means forced color => only that color is possible.
        # Adjacent constraint => dp[i][c] can come from dp[i-1][c'] for c' != c.
        dp_prev = [0,0,0]  # ways for i-1
        # initialize for i=0
        if line[0] == -1:
            dp_prev = [1,1,1]  # can choose color 0,1,2
        else:
            dp_prev = [0,0,0]
            dp_prev[line[0]] = 1
        
        for i in range(1, n):
            dp_cur = [0,0,0]
            if line[i] == -1:
                # can choose any of {0,1,2} but must differ from line[i-1]
                for c in range(3):
                    # sum over c' != c
                    s = 0
                    for c2 in range(3):
                        if c2 != c:
                            s += dp_prev[c2]
                    dp_cur[c] = s % MOD
            else:
                c = line[i]
                # forced color c
                s = 0
                for c2 in range(3):
                    if c2 != c:
                        s += dp_prev[c2]
                dp_cur[c] = s % MOD
            dp_prev = dp_cur
        
        ans = sum(dp_prev) % MOD
        print(ans)
        return

    # ----------------------------------------------------------------
    # Otherwise, both H>1 and W>1, but H*W <= 200.  We will do a row-based (or column-based)
    # DP using a standard "paint by rows" approach, exploiting that adjacent cells in
    # the same row cannot share the same color, and similarly between rows.
    #
    # Key optimization: We take the smaller dimension as "number of rows" in the DP,
    # to keep the "width" dimension as the row length.  Then we iterate over those "rows."
    # ----------------------------------------------------------------

    # If H > W, transpose the grid so that R = min(H,W)
    # That way, R = number of "rows" in our DP, C = length of each row
    # Then after DP we revert.  Actually, we only need to do the DP on the transposed forced[].
    if H <= W:
        R, C = H, W
        transposed_forced = forced
    else:
        R, C = W, H
        # transpose forced
        transposed_forced = [[forced[r][c] for r in range(H)] for c in range(W)]

    # We'll work with transposed_forced as a grid of size R x C
    # transposed_forced[r][c] is the forced color (0..2 or -1).
    # We must ensure adjacency constraints within a row r (i.e. c and c+1),
    # and between row r and r+1 at column c.

    # ----------------------------------------------------------------
    # Precompute all valid patterns for each row r ignoring adjacency to any other row,
    # but respecting forced constraints in that row and "within-row" adjacency (no two adjacent same).
    #
    # We'll store each pattern as an integer up to 2*C bits (though we only use 0..2 for each column).
    # Then row_valid[r] is the list of such pattern-ints, row_valid_map[r][pattern_int] = index.
    # ----------------------------------------------------------------

    # Encode color array [c0, c1, ...] of length C (each c in {0,1,2}) into an integer with 2 bits per column.
    # We'll store color c in bits 2*c..2*c+1.  That means we can safely store up to c=3, but we'll only use c=0..2.

    def build_valid_row_patterns(row_forced):
        # row_forced is array of length C with forced color or -1
        # generate all sequences col[0..C-1] in {0,1,2} subject to
        #  - if row_forced[c]>=0, must use that color
        #  - no two adjacent columns the same
        valid_patterns = []
        # We'll do a DFS/ backtrack in an iterative manner
        stack = [(0, -1, 0)]  # (col, prev_color, pattern_int_so_far)
        # pattern_int_so_far will store colors for columns [0..col-1].
        # prev_color is color assigned to column col-1, or -1 if col=0.

        while stack:
            col, prev_col_color, pat = stack.pop()
            if col == C:
                # we've assigned all columns
                valid_patterns.append(pat)
                continue

            fc = row_forced[col]
            if fc != -1:
                # forced color
                if fc == prev_col_color:
                    # invalid
                    continue
                # encode fc in bits 2*col..2*col+1
                new_pat = pat | (fc << (2*col))
                stack.append((col+1, fc, new_pat))
            else:
                # can choose any color 0..2 except prev_col_color
                for ccol in range(3):
                    if ccol != prev_col_color:
                        new_pat = pat | (ccol << (2*col))
                        stack.append((col+1, ccol, new_pat))
        return valid_patterns

    row_valid = []
    row_valid_map = []
    for r in range(R):
        row_force = transposed_forced[r]
        vp = build_valid_row_patterns(row_force)
        vp.sort()  # not strictly necessary, but easy to keep consistent ordering
        mp = {}
        for i, p in enumerate(vp):
            mp[p] = i
        row_valid.append(vp)
        row_valid_map.append(mp)
        if len(vp) == 0:
            # no valid way to color this row
            print(0)
            return

    # dp[r][idx_of_pattern_in_row_valid[r]] = number of ways to color up to row r
    # We'll do a rolling 2-row approach to save memory.
    dp_prev = [0]*len(row_valid[0])
    for i in range(len(row_valid[0])):
        dp_prev[i] = 1  # any pattern in row 0 is possible (already forced-check ok)

    # ----------------------------------------------------------------
    # For row r => r+1 transitions, we must ensure at each column c:
    #   color(r,c) != color(r+1,c).
    # We'll build these transitions on the fly (rather than a giant adjacency table)
    # to save memory.  For each pattern p in row r, we systematically generate
    # all patterns q in row r+1 that are compatible (i.e. differ in each column)
    # and also respect "within-row" adjacency in r+1 (but that is already guaranteed
    # in row_valid[r+1]) and forced constraints (also guaranteed by row_valid[r+1]).
    #
    # However, we must still check column-by-column p[c] != q[c].
    # Instead of enumerating all of row_valid[r+1], we "rebuild" those patterns
    # by a small DFS that enforces q[c] != p[c].  But that would re-generate them
    # from scratch.  Another approach is to simply iterate over each q in row_valid[r+1]
    # and compare with p.  That is len(row_valid[r+1]) up to ~3*2^(C-1).  If both lists
    # can be large, that can be big.  But we will do it in a controlled manner below.
    #
    # An alternate approach is indeed to do a quick check for each q in row_valid[r+1]:
    #   for c in 0..C-1: if (p>>2c & 3) == (q>>2c & 3), conflict => break
    #   if not conflict => dp[r+1][q_id] += dp[r][p_id]
    # This is simpler to implement. We just have to be sure we don't time out.
    #
    # Since R*C <= 200, the maximum C is <= 200. But if both dimensions > 1,
    # then the smaller dimension is at most ~14 (since 14*14=196). For C=14,
    # row_valid[r] can have up to 3*2^(13)=24576 patterns.  Checking adjacency
    # pairwise is 24576^2=6.0e8 which is quite large in Python. 
    #
    # A more efficient way: For each p, we can do a direct DFS to build q-rows that differ
    # in each column c. That is up to 2^C=16384 expansions, times 24576=4e8 - still large
    # but possibly somewhat faster in practice because of early break etc. It is borderline
    # in Python. 
    #
    # Given the problem constraints, we try to implement the "check each q" approach
    # but skipping those dp[r][p_id] == 0, and also if row_valid[r+1] is large, it may
    # be big. We will do our best to optimize in Python. Many test cases will have forced
    # constraints that reduce the pattern counts drastically in practice.  We will implement
    # the direct "compare bits" approach below, which is typically simpler.
    # ----------------------------------------------------------------

    # A small helper to check if p and q differ in all columns
    def compatible_patterns(p, q, ccount):
        # ccount = C
        # check p[c] != q[c] for c in 0..ccount-1
        # p[c] = (p >> (2*c)) & 3
        # q[c] = (q >> (2*c)) & 3
        # we only use colors 0..2, but we stored them in 2 bits
        # do a quick loop
        for i in range(ccount):
            pc = (p >> (2*i)) & 3
            qc = (q >> (2*i)) & 3
            if pc == qc:
                return False
        return True

    import math

    # Prepare DP array for row r+1
    for r in range(R-1):
        dp_cur = [0]*len(row_valid[r+1])
        # only iterate over p_id where dp_prev[p_id] != 0
        for p_id, ways in enumerate(dp_prev):
            if ways == 0:
                continue
            p = row_valid[r][p_id]
            # check each q in row_valid[r+1]
            for q_id, q in enumerate(row_valid[r+1]):
                # check if p, q differ in all columns
                # This is the adjacency check
                # If differ => dp_cur[q_id] += ways
                if compatible_patterns(p, q, C):
                    dp_cur[q_id] = (dp_cur[q_id] + ways) % MOD
        dp_prev = dp_cur

    ans = sum(dp_prev) % MOD
    print(ans)