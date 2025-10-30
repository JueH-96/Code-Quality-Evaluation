def main():
    import sys
    sys.setrecursionlimit(10**7)
    MOD = 998244353
    
    input_data = sys.stdin.read().strip().split()
    H = int(input_data[0])
    W = int(input_data[1])
    grid_lines = input_data[2:]
    
    # We have a grid S of size H x W.  Each cell is one of '1','2','3','?'.
    # We want to count how many ways to replace '?' by one of {1,2,3}
    # so that no two edge-adjacent cells have the same digit.
    #
    # Constraints:
    #  - 1 <= H, W
    #  - H*W <= 200
    #  - Grid characters in { '1','2','3','?' }
    #
    # We must output the result modulo 998244353.
    #
    # ---------------------------------------------------------
    # Key observations / approach:
    #
    # 1) Immediately check for conflicts among the already-fixed cells:
    #    If two adjacent cells in the grid are both fixed and have the same digit,
    #    then the answer is 0.
    # 2) Each '?' cell can only be assigned digits that do not conflict
    #    with its fixed neighbors.  If a '?' has a neighbor with digit X,
    #    then that '?' cannot be X.  If that removes all possible digits,
    #    answer is 0.
    # 3) Among the '?' cells themselves, we must ensure no two adjacent
    #    '?'s get the same digit.
    #
    # Because H*W <= 200, we can afford algorithms up to roughly O(3^(H+W)) if
    # H or W is small, or we can use a graph-based approach.  However, a direct
    # 3^q enumeration (where q is number of '?') could be up to 3^200, which is intractable.
    #
    # But note this is a standard "proper 3-coloring" problem on a grid-graph
    # (bipartite), with some vertices' color pre-fixed or partially restricted.
    # Counting the number of 3-colorings of a bipartite graph is still nontrivial
    # in general.  But we exploit that the grid is not too large in both dimensions
    # at once because H*W <= 200.  We can do a row-by-row or column-by-column DP
    # if min(H,W) is small enough.
    #
    # Indeed, if we pick the larger dimension to be "rows" in a DP sense, we face 3^W states
    # if we do row-based DP.  That could be too large if W=200.  But if W=200,
    # then H=1, so the counting is trivial in that case (a single row).
    # Similarly if H=200, then W=1, also trivial.  If both H and W are at least 2,
    # then neither can exceed 14 (since 15*15=225 > 200).  In that scenario,
    # W<=14 or H<=14.  3^14 = about 4.78e6, which may be borderline but still doable
    # with efficient implementation in Python if done carefully (and using fast I/O).
    #
    # We'll implement a row-by-row DP where "state" is the coloring of the previous row.
    # Then we try all valid colorings of the next row (respecting horizontal adjacency
    # and the pre-fixed cells, or '?'-supported color sets) and also vertical adjacency
    # with the previous row.  We sum the number of ways modulo 998244353.
    #
    # Steps:
    #  A) First store the grid in a 2D array.  Check adjacency of fixed cells:
    #     For each edge, if both endpoints are fixed to the same digit => 0.
    #  B) Build a list of allowed colors for each cell.  If it's '1','2','3',
    #     then that cell has exactly 1 allowed color.  If it's '?', initially it has {1,2,3},
    #     but we might remove colors that are disallowed because of adjacency to a fixed cell
    #     with that color.  If any cell ends up with zero allowed colors => 0.
    #  C) Choose orientation of DP: If H <= W, we'll use row-based DP (the row length is W).
    #     Otherwise, we can transpose the grid and do DP by columns (the "width" becomes H).
    #     So let R = min(H, W) as the "DP dimension".  We'll treat the dimension R
    #     as the number of cells in each "row" of DP.  Actually we'll keep everything consistent:
    #        - If H <= W, we'll do row-based DP with H rows and W columns.
    #        - If H > W, we'll transpose the grid so we do DP with W "rows" and H "columns".
    #     This ensures the "width" we do a DP over is always the smaller dimension,
    #     so we have at most 14 or so columns to handle in the worst case.
    #
    #  D) Then for each row i, we define a "pattern" for that row as an array of length W
    #     of digits in {1,2,3}, with the constraint that horizontally adjacent cells differ,
    #     and each cell's color is in its allowed set.  We'll gather all such valid patterns
    #     for row i (there can't be more than 3^W patterns in the worst case; with adjacency
    #     it can be fewer).
    #  E) We'll maintain a DP that for each row i, and each valid pattern p_i, how many ways
    #     to reach that pattern.  Then for the next row i+1, we try all valid patterns p_{i+1}
    #     that do not conflict vertically with p_i.  (Meaning, for each column j, the color
    #     p_i[j] != p_{i+1}[j].)  Then we do DP[i+1, p_{i+1}] += DP[i, p_i].
    #  F) At the end, sum up DP[H-1, p] for all valid patterns p to get the final count.
    #
    # Implementation details:
    #
    #  - We'll possibly transpose the grid if W > H, so that we always do DP over
    #    "rows" = the larger dimension, "cols" = the smaller dimension.  Actually we want
    #    the "cols" dimension to be the smaller one for the pattern-based DP.
    #    So let's define:
    #      if W <= H: (row_count, col_count) = (H, W)
    #         grid as is
    #      else: (row_count, col_count) = (W, H)
    #         transpose the grid
    #  - Then we enumerate valid color assignments for a single row i as a list of length col_count.
    #    For each column c in [0..col_count-1], the cell is (i,c) in the transposed sense.
    #    That cell must pick a color from its allowed set, and no two adjacent columns can share color.
    #    We'll store all such row-patterns in a list rowPatterns[i].
    #    Actually the set of possible patterns might differ from row to row if different cells
    #    have different allowed color sets.  So we must generate them row by row.
    #
    #  - Then we build adjacency info across consecutive rows i and i+1.  When we choose a pattern
    #    p_i for row i and a pattern p_{i+1} for row i+1, the only vertical adjacency constraints
    #    are that p_i[c] != p_{i+1}[c] for each c.  So to do DP transitions, we just check
    #    for all c, p_i[c] != p_{i+1}[c].
    #
    #  - We'll do a standard DP approach:
    #       DP[i][idx] = sum of DP[i-1][prev_idx] for all prev_idx that is compatible
    #       with current pattern idx.
    #    We'll keep these arrays modded by 998244353.
    #
    # Complexity:
    #  - For each row, we generate all valid patterns.  Each pattern is at most 3^col_count in naive approach,
    #    but we must filter out patterns that have adjacent columns with the same color or that violate the allowed set.
    #    This is 3^col_count <= 3^14 ~ 4.78e6 max if col_count=14, which might be borderline in Python.
    #    We can reduce it by noting that adjacent columns can't share color, so the number of valid "no two same adjacent colors"
    #    sequences of length col_count over 3 colors is 3 * 2^(col_count-1) = at most 3*2^13=3*8192=24576 if col_count=14,
    #    which is much smaller (~2.5e4).  Then we also must factor in the allowed sets, which might reduce it further,
    #    but let's assume worst case everyone is '?' so each cell can be any color.  Then the only constraint horizontally
    #    is "adjacent columns differ."  That yields 3*2^(col_count-1) patterns, which is ~24k for col_count=14.  That's still
    #    potentially large but more manageable than 4.7e6.  For col_count=14 and row_count=14, we have to handle 14 rows.
    #
    #  - Then we need to build a compatibility table between row i's patterns and row i+1's patterns.  Each row can have up to
    #    ~24k patterns in the worst case.  Checking compatibility means p_i[c] != p_{i+1}[c] for each c in [0..col_count-1].
    #    We can do that in O(col_count) time per pair.  So that might be 24k * 24k * 14 => ~8.07e9 operations if done naively,
    #    which is quite large for Python.  We need to optimize more carefully by building a hash or bit-based representation.
    #
    #    However, we can store each pattern as an integer in base 3 or store it in bits with 2 bits per cell.  Checking adjacency
    #    can be done quickly.  But 8e9 is probably too big for Python in typical constraints.  We'll need a more efficient approach.
    #
    #  - A known optimization is to store each pattern as a string/tuple of length col_count, or an integer encoding "p_0 + p_1*3 + ... p_{col_count-1}*3^(col_count-1)".  Then to check if two patterns differ in each column, we can do a fast bitwise technique if we compress the colors with two bits each.  Then "different in each position" is a check that for no position do they match.  We can do that by a couple of bitwise tricks, but we have to be careful.  Alternatively, we can do a simpler approach: we can store each pattern in a Python integer with 2 bits per column => up to 28 bits for col_count=14, which fits in 32-bit.  Then to check if patternA and patternB differ in all 14 columns, we can compute "eq = ~(patternA ^ patternB)" & a mask that picks out each pair of bits if they are the same.  If eq != 0, then there's at least one column that is the same color.  Actually, we want to check they differ in each column?  Wait, we want to ensure p_i[c] != p_{i+1}[c].  That means for each c, color_i[c] != color_{i+1}[c].  So we want no column c with color_i[c] == color_{i+1}[c].  So patternA and patternB must differ in each pair of bits for each column.
    #
    #    If we encode color 1->0, 2->1, 3->2 for instance (only 3 possible values, we can store in 2 bits).  Then to check if colorA==colorB, we compare those 2 bits.  We want to ensure for every column c, colorA[c] != colorB[c].  That means for no c do the 2 bits match.  We can do a bitwise approach if we store the entire row pattern in 2 bits per column, so col_count=14 => 28 bits.  Then a single bitwise XOR is 28 bits.  If color_i[c] == color_{i+1}[c], then the 2 bits for that column are identical => XOR = 00.  So we want to ensure that for all columns c, we do not get 00 in those 2 bits of the XOR.  i.e. we want the XOR to have no group-of-2-bits == 00.  Checking that is more complicated than just eq=0.  Because if colorA=1 (encoded 0) and colorB=1 (encoded 0), the XOR is 0 in that pair of bits, so we do not want that.  We need a quick method to check that no pair of bits in the XOR is 00.  
    #
    #    We can do it in a small loop or by precomputing a table of patternA-> set of patterns in B that differ in all columns.  We'll store "DiffAll" compatibility in a dictionary.  Because the number of distinct patterns is up to ~24k, we can store for each pattern a bit.  Then building a full adjacency might be 24k^2 ~ 576 million in the worst case.  Might still be too large for Python.  We can do additional pruning though, or we can do a smarter trick: Because we only require that each column differs, we can do a per-column approach with a bitmask or something.  We'll see if we can optimize enough in Python with careful data structures.
    #
    #    Alternatively, we can do a forward DP: for row i+1, we build sums over which "color configuration" row i used in each column.  But that again might lead to large overhead.
    #
    #    Another partial optimization is that for each row i, the set of patterns might be significantly smaller than 3*2^(col_count-1) due to some cells having restricted sets (like if a cell can only be color '1').  Also in big examples, typically not all cells are '?'; but the problem statement does allow many '?'. 
    #
    # We'll implement the row-by-row DP with the following plan:
    #
    #   Step 1: Possibly transpose if W < H so that the "width" for patterns is min(H, W).
    #   Step 2: For each row i, build a list of valid row-patterns P[i], and store them in compressed form (an integer up to 2*(col_count) bits).  We'll also keep an integer "patternID" for each valid pattern, so P[i] is a list of (pattern_bits, pattern_id).
    #   Step 3: We'll create a mapping from pattern_bits to pattern_id.  We'll do this row by row because the allowed sets can differ from row to row.  So each row has its own pattern -> ID map.
    #   Step 4: We'll build a "compatibility" structure: for each pattern_id in row i, produce a list of pattern_id's in row i+1 that differ in every column.  We'll do this row-by-row. 
    #       - We'll generate P[i+1] first, then for each pattern in P[i], we check compatibility with each pattern in P[i+1] in O(#P[i+1]) time.  This might still be large, but we'll attempt to implement it carefully. 
    #   Step 5: We run a DP: dp[i][id] = number of ways to color up to row i with row i using pattern with ID = id.  Then dp[i+1][id2] += dp[i][id1] for all id2 in compat[i][id1].
    #
    # Implementing the "differ in every column" check:
    #   - We'll have pattern_bits_i (up to 2 * col_count bits).  We'll do a function "all_diff_in_columns(bitsA, bitsB, col_count)" that returns True if for every column c, the 2-bit color code in bitsA for column c != the code in bitsB for column c.  We'll implement that quickly in a small loop of col_count or some bitwise trick.
    #   - Because col_count <= 14 in the worst case, we can do it in a small unrolled loop.  It's 14 checks max, which might be feasible if we do it 24k * 24k times => ~576 million.  That might be borderline in Python, but might be optimized enough with pypy or fast methods if carefully done.  We have to be mindful of time.
    #   - We can attempt a faster method by precomputing for each possible 28-bit pattern, a 28-bit mask that indicates which pattern of bits would yield a conflict.  But that might be complicated.  Another approach is partial caching or hashing.  
    #
    #   We'll try to implement it directly but in a more optimized manner.  If it seems too big, we might do a more advanced approach using an approach column by column with a tries structure.  But let's first do the straightforward approach with potential early break, and we store the result in adjacency lists.
    #
    # Implementation details:
    #
    # (A) Input reading and building "allowedColors[r][c]" as a set of color indices in {0,1,2} meaning {1,2,3}:
    #     We'll use color 0 for digit '1', 1 for '2', 2 for '3'.
    #
    # (B) Check conflict among fixed neighbors => if conflict => print(0) and return.
    #
    # (C) Possibly transpose if that yields fewer columns.  Keep track of new H, W accordingly.
    #
    # (D) For each row, generate all valid patterns that (1) respect each cell's allowed color set
    #     (2) no two horizontally adjacent cells share the same color.  Store them as an integer pattern_bits.
    #
    # (E) Build adjacency/compatibility between row i's patterns and row i+1's patterns (they must differ in each column).
    #
    # (F) DP from row 0 to row H-1 and sum results.
    #
    # Edge cases: (H=1 and W=1 => trivially 1 or up to 3 if '?', as long as no conflict.)
    # We'll handle them in the same DP logic, it's just a single row.
    #
    # We'll code carefully for performance.
    #
    # Let's implement now.
    
    # -------------------------------------------------------------
    # Step 1: Parse the grid into a 2D array of characters.
    S = [list(row) for row in grid_lines]  # S[i][j]
    
    # Step 2: Check adjacency of fixed cells => immediate conflict => 0
    # Also build an initial "allowed colors" for each cell as a bitmask in range 0..7 (3 bits => color0,1,2).
    # color '1' => bit 1 (decimal 1), color '2' => bit 2, color '3' => bit 4.
    
    # We'll define a helper to convert a character to a bitmask of allowed colors:
    def char_to_mask(ch):
        if ch == '1': return 1  # 001 in binary
        if ch == '2': return 2  # 010 in binary
        if ch == '3': return 4  # 100 in binary
        if ch == '?': return 7  # 111 in binary (all 3 allowed)
        # Should never happen
        return 0
    
    allowed = [[char_to_mask(S[i][j]) for j in range(W)] for i in range(H)]
    
    # Function to get the integer color from a single-bit set (like 1 -> color0, 2 -> color1, 4-> color2)
    # but note that some cells might have multiple bits set if they're '?' and not restricted yet.
    # For conflict checking, we only need to see if a cell is fixed to a single color or not.
    # If it has exactly one bit set, we can get that color by index_of_bit(allowed[i][j]).
    def single_color_index(mask):
        # mask is one of {1,2,4}, we want to return 0 if mask==1, 1 if mask==2, 2 if mask==4
        if mask == 1: return 0
        elif mask == 2: return 1
        elif mask == 4: return 2
        else: return -1  # not a single color
    
    # Directions for adjacency (up,down,left,right)
    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    
    for i in range(H):
        for j in range(W):
            c1 = single_color_index(allowed[i][j])
            if c1 == -1:
                # not a single fixed color, skip
                continue
            # c1 is the color index for S[i][j].
            # check neighbors
            for di, dj in directions:
                ni, nj = i+di, j+dj
                if 0 <= ni < H and 0 <= nj < W:
                    c2_mask = allowed[ni][nj]
                    # If c2_mask is single and c1==c2 => conflict => 0
                    c2 = single_color_index(c2_mask)
                    if c2 == c1 and c2 != -1:
                        print(0)
                        return
                    # else remove c1 bit from c2_mask if c2_mask is not single
                    # Actually we should remove color c1 from neighbor's allowed set
                    # because they can't be the same as c1 if they're adjacent.
                    # But only if the neighbor is '?'.  Actually we must do it for any neighbor
                    # that is not already single with a different color.
                    new_mask = c2_mask & ~(1 << c1)  # remove the bit 1<<c1
                    if new_mask == 0:
                        # no way to color neighbor => 0
                        print(0)
                        return
                    allowed[ni][nj] = new_mask
    
    # Now we have pruned the allowed sets based on adjacency to fixed cells.
    # Next step: if any cell is single-colored, we can propagate that restriction further
    # because that also disallows that color in adjacent cells.  We can do a queue-based
    # iterative propagation until stable.
    
    from collections import deque
    q = deque()
    # Initialize queue with all single-colored cells
    for i in range(H):
        for j in range(W):
            if bin(allowed[i][j]).count('1') == 1:
                q.append((i,j))
    
    while q:
        i, j = q.popleft()
        color_mask = allowed[i][j]
        cidx = single_color_index(color_mask)
        if cidx == -1:
            continue
        # cidx is the single color used by cell (i,j)
        for di, dj in directions:
            ni, nj = i+di, j+dj
            if 0 <= ni < H and 0 <= nj < W:
                before = allowed[ni][nj]
                # remove cidx
                after = before & ~(1 << cidx)
                if after == 0:
                    print(0)
                    return
                if after != before:
                    allowed[ni][nj] = after
                    if bin(after).count('1') == 1:
                        q.append((ni,nj))
    
    # If we get here, no immediate contradictions.  We must now count ways
    # to assign a color from allowed[i][j] to each cell so that no two adjacent
    # '?' cells share the same color.  But more simply, it's that no two adjacent
    # cells share color, but we've already pruned adjacency with fixed cells.
    # We still must ensure no two adjacent '?' cells pick the same color among
    # their allowed sets.  We'll do the row-by-row DP approach.
    
    # Decide if we transpose:
    # We'll define row_count = H, col_count = W.  If col_count > row_count, we transpose.
    # This ensures col_count <= row_count, so col_count = min(H, W).
    
    if W > H:
        # transpose
        row_count, col_count = W, H
        new_allowed = [[allowed[r][c] for r in range(H)] for c in range(W)]
    else:
        row_count, col_count = H, W
        new_allowed = allowed
    
    # Now new_allowed[r][c] is the allowed bitmask for row r, column c, with col_count <= row_count.
    # We'll build all valid patterns for each row r.
    
    # Precompute all valid row patterns row by row.
    # A "row pattern" is an assignment of each column c in [0..col_count-1] to exactly one color
    # from that cell's allowed set, such that no two adjacent columns have the same color.
    #
    # We'll store each pattern as an integer with 2 bits per column c.  color can be in [0..2].
    # We'll define an encoding color 0->bits 00, color 1->bits 01, color 2->bits 10 (or some minimal scheme),
    # but we only need to store up to 2 bits since we have 3 possible color values.  We'll just store them as 0,1,2 in a base-4 or base-3. 
    # Actually we can store them in base 4 for convenience: color c -> c in [0..2], keep 2 bits.  The top bit combination 3 won't be used.
    #
    # We'll build a list: rowPatterns[r] = list of (pattern_bits, count_of_ways?), but we actually only need pattern_bits.
    # Then we'll define a map rowMap[r][pattern_bits] -> index in rowPatterns[r], so we can quickly find pattern IDs.
    
    # Let's define a function to generate all valid patterns for a single row of length col_count,
    # given new_allowed[r][c].
    
    def generate_row_patterns(r):
        # We'll do a DFS or iterative approach to build all sequences of length col_count,
        # each element is color in [0..2], subject to color in allowed bitmask for that cell,
        # and color differs from the previous color.  col_count <= 14, so we can do a backtracking
        # up to 14 columns.  The maximum number of valid patterns is 3*2^(col_count-1) if all are '?'.
        # We'll store each pattern as an integer using 2 bits per color in base 4 encoding.
        
        results = []
        
        def backtrack(c, prev_color, pattern_bits):
            if c == col_count:
                results.append(pattern_bits)
                return
            mask = new_allowed[r][c]
            # try each color in [0..2] that is allowed by mask
            # and differs from prev_color
            for color in range(3):
                if (mask & (1 << color)) != 0:
                    if color != prev_color:  # ensure no horizontal conflict
                        # encode color in 2 bits: color
                        # pattern_bits so far is for columns [0..c-1].
                        # shift by 2 bits, add color
                        new_bits = (pattern_bits << 2) | color
                        backtrack(c+1, color, new_bits)
        
        backtrack(0, -1, 0)
        return results
    
    rowPatterns = []
    for r in range(row_count):
        pats = generate_row_patterns(r)
        rowPatterns.append(pats)
    
    # Next, build a map from pattern_bits to index, for each row r.
    # We'll store them in a dictionary for fast lookup.
    rowMap = []
    for r in range(row_count):
        mp = {}
        for idx, bits in enumerate(rowPatterns[r]):
            mp[bits] = idx
        rowMap.append(mp)
    
    # Now build adjacency lists "compat[r]" that for each pattern index in row r
    # gives the list of pattern indices in row r+1 that are vertically compatible.
    # Two patterns are vertically compatible if at each column c, the color in row r,
    # col c differs from row r+1, col c.  We have them stored in 2-bits per column,
    # so we need to decode or directly check them.
    #
    # We'll build a helper that, given pattern_bits for row r and for row r+1,
    # checks if they differ in each column.
    
    def patterns_vert_compatible(bitsA, bitsB):
        # We have col_count columns, each color stored in 2 bits.  We'll extract them from right to left.
        # If for any column the color is the same => return False, else True.
        # col_count <= 14, so let's just do a small loop.
        for _ in range(col_count):
            cA = bitsA & 3
            cB = bitsB & 3
            if cA == cB:
                return False
            bitsA >>= 2
            bitsB >>= 2
        return True
    
    compat = []
    for r in range(row_count-1):
        cur_list = rowPatterns[r]
        nxt_list = rowPatterns[r+1]
        adj = [[] for _ in range(len(cur_list))]
        # We'll attempt a more efficient method than naive O(len(cur_list)*len(nxt_list)*col_count).
        # But let's do it straightforward first. We'll see if we must optimize.
        # length can be up to ~3*2^(col_count-1). For col_count=14, that's up to 24576 patterns per row.
        # so worst-case is 24576^2 ~ 6e8 checks for all pairs from row r to r+1. That might be borderline.
        # We'll implement partial optimizations: short-circuit once we find any same color in a column.
        
        # A possible approach: we can bucket next_list by a signature of the pattern. But let's just do nested loops,
        # with a short-circuit. We'll try to do it in C++-like style in Python. It's still risky in time.
        
        # Implementation detail: let's store them in local arrays to accelerate attribute lookup.
        for i, ba in enumerate(cur_list):
            for j, bb in enumerate(nxt_list):
                if patterns_vert_compatible(ba, bb):
                    adj[i].append(j)
        compat.append(adj)
    
    # If row_count=1, there's no "compat" to build. We'll handle that as a special case in DP.
    
    # Now we do the DP:
    # dp[r][i] = number of ways to color up to row r with pattern i in row r.
    # We'll do a rolling array approach to save memory, dp_cur, dp_next.
    
    import math
    
    if row_count == 1:
        # Then the answer is just the number of valid patterns for row 0,
        # because there's no vertical adjacency to worry about beyond that.
        ans = len(rowPatterns[0]) % MOD
        print(ans)
        return
    else:
        dp_prev = [0]*len(rowPatterns[0])
        # initial row: each pattern is 1 way
        for i in range(len(rowPatterns[0])):
            dp_prev[i] = 1
        for r in range(row_count-1):
            dp_next = [0]*len(rowPatterns[r+1])
            adj = compat[r]  # adjacency from row r to row r+1
            for i in range(len(rowPatterns[r])):
                ways_i = dp_prev[i]
                if ways_i != 0:
                    for j in adj[i]:
                        dp_next[j] = (dp_next[j] + ways_i) % MOD
            dp_prev = dp_next
        ans = sum(dp_prev) % MOD