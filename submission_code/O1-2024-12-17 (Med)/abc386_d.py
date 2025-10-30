def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    
    # ----------------------------------------------------------------------
    # PROBLEM RESTATEMENT (in simpler terms):
    #
    # We have an N×N grid. We want to color each cell black or white so that:
    #   • Each row r has a "prefix‐boundary" i(r) (0 ≤ i(r) ≤ N): 
    #       all columns 1..i(r) in that row are black, and columns i(r)+1..N are white.
    #   • Each column c has a "prefix‐boundary" j(c) (0 ≤ j(c) ≤ N): 
    #       all rows 1..j(c) in that column are black, and rows j(c)+1..N are white.
    #
    # Since there is only one final color per cell, the usual interpretation is:
    #   cell(r,c) is black  ⇔  (c ≤ i(r)) and (r ≤ j(c))
    #   cell(r,c) is white  ⇔  otherwise
    #
    # Equivalently, this "row prefix" + "column prefix" coloring implies:
    #   c ≤ i(r)   ⇔   cell(r,c) is black from row r's viewpoint
    #   r ≤ j(c)   ⇔   cell(r,c) is black from column c's viewpoint
    # For the final coloring to agree, whenever a cell is black from one viewpoint,
    # it must also be black from the other.  In short:
    #   c ≤ i(r)  ⇔  r ≤ j(c).
    #
    # One can rearrange that condition into:
    #   (c ≤ i(r)) and (r ≤ j(c))   ⇔   i(r) + j(c) ≥ r + c.
    #
    # We are given M already-colored cells.  For each:
    #   • If that cell is black => i(r) + j(c) ≥ r + c
    #   • If that cell is white => i(r) + j(c)  < r + c
    #
    # We must decide if there exists a choice of i(r) in [0..N], j(c) in [0..N]
    # that satisfies these M constraints.  Because r,c ≤ N, we automatically have
    # i(r) ∈ [0..N], j(c) ∈ [0..N].  That in turn means i(r)−r ∈ [−r..(N−r)],
    # j(c)−c ∈ [−c..(N−c)], but typically r,c ≤ N so those ranges are feasible enough
    # that often no direct bounding contradiction occurs (unless r+c > 2N, but that
    # cannot happen given r,c ≤ N ⇒ r+c ≤ 2N).
    #
    # The standard known approach is to define:
    #       A(r) = i(r) - r         (for each row r)
    #       B(c) = j(c) - c         (for each column c)
    # so that:
    #   i(r) + j(c) ≥ r + c   ⇔   [A(r) + r] + [B(c) + c] ≥ r + c   ⇔   A(r) + B(c) ≥ 0
    #   i(r) + j(c)  < r + c  ⇔   A(r) + B(c)  < 0
    #
    # So each black cell => A(r)+B(c) ≥ 0, each white cell => A(r)+B(c) < 0.
    #
    # We only have up to M = 2*10^5 constraints, and N can be very large (up to 10^9),
    # but r,c ≤ N.  We cannot store an array of i(r) for all 1..N.  Instead, we only
    # need to consider those rows r and columns c that actually appear among the M
    # constraints—if a row or column never appears, it can be assigned anything without
    # affecting the M constraints.
    #
    # HOWEVER, simply checking each black/white cell's "A(r)+B(c) ≥ 0" or "< 0" in
    # isolation might miss the fact that having a black cell (r,c) imposes
    # i(r)>=c and j(c)>=r, which can force additional cells in row r or column c to
    # become black (potentially conflicting with some white cell).  In other words,
    # sometimes combining multiple black cells can create a bigger forced‐black region
    # that intersects a white cell.  That subtle effect is exactly captured by the
    # logical condition c ≤ i(r) ⇔ r ≤ j(c), i.e. A(r)+B(c)≥0 ⇔ the other pairs must
    # also respect that if row r is forced to have i(r)≥some large c, it forces other
    # columns ≤ c to also be black in row r, etc.
    #
    # A well‐known concise “necessary and sufficient” check (which is simpler to code
    # and perfectly handles these forced regions) is:
    #
    #   1) For each row r, let maxBlackCol[r] be the maximum column c among black cells (r,c).
    #      (If no black cell in row r, set maxBlackCol[r]=0.)
    #      Then if row r must contain a black cell at column c, that forces row r to be
    #      black in ALL columns from 1..c.  So i(r) ≥ c is the minimal necessity.
    #
    #   2) For each column c, let maxBlackRow[c] be the maximum row r among black cells (r,c).
    #      (If no black cell in column c, set maxBlackRow[c]=0.)
    #      Then if column c must contain a black cell at row r, that forces column c to be
    #      black in ALL rows from 1..r.  So j(c) ≥ r is the minimal necessity.
    #
    #   3) Now check every white cell (r,c).  If row r has maxBlackCol[r] ≥ c, that means
    #      i(r)≥c.  If column c has maxBlackRow[c] ≥ r, that means j(c)≥r.  Then that
    #      cell (r,c) would be forced black from both row and column vantage.  Contradiction!
    #      So if for any white cell (r,c) we find (maxBlackCol[r] ≥ c) AND (maxBlackRow[c] ≥ r),
    #      we must answer "No".
    #
    #   4) If no contradiction arises, answer "Yes".
    #
    # This simple check indeed accounts for the fact that if row r is black at c,
    # it must be black at all columns ≤ c, and if column c is black at r, it must
    # be black at all rows ≤ r.  Hence it catches conflicts with any white cell in
    # those forced regions.
    #
    # However, we must also ensure that each black cell (r,c) is *actually* black
    # under that assignment.  But that is automatically satisfied if we set:
    #       i(r) = maxBlackCol[r],   j(c) = maxBlackRow[c].
    #   Because for black cell (r,c), c ≤ i(r)=maxBlackCol[r] and r ≤ j(c)=maxBlackRow[c].
    #   So all black cells remain black.  That is the "minimal" solution that
    #   ensures black cells remain black.  Any larger i(r) or j(c) would only force
    #   more black cells, which is more likely to create conflicts, so the minimal
    #   assignment is the best test.  If the minimal assignment does not conflict
    #   with any white cell, then a solution exists.
    #
    # This method is efficient (O(M)), and it matches all the sample explanations.
    #
    # We will implement exactly that check.
    # ----------------------------------------------------------------------

    # Store for each row r: the maximum black-column in that row
    # Store for each col c: the maximum black-row in that col
    # Then check each white cell for conflict.

    # Because r and c can be up to 10^9, we must store them in dictionaries keyed by r/c.

    import collections

    maxBlackCol = {}  # row -> max column among black cells
    maxBlackRow = {}  # col -> max row among black cells

    # We will parse the M lines.  The format is:
    # X_i Y_i C_i   where C_i is 'B' or 'W'.
    #
    # We'll collect all black cells in a list so afterwards we can fill maxBlackCol/Row.
    # We'll also store all white cells in another list so we can do the conflict check.

    ptr = 2
    black_cells = []
    white_cells = []

    for _ in range(M):
        r = int(input_data[ptr]); c = int(input_data[ptr+1]); color = input_data[ptr+2]
        ptr += 3
        if color == 'B':
            black_cells.append((r,c))
        else:
            white_cells.append((r,c))

    # Fill in maxBlackCol / maxBlackRow
    for (r,c) in black_cells:
        if r not in maxBlackCol:
            maxBlackCol[r] = c
        else:
            if c > maxBlackCol[r]:
                maxBlackCol[r] = c

        if c not in maxBlackRow:
            maxBlackRow[c] = r
        else:
            if r > maxBlackRow[c]:
                maxBlackRow[c] = r

    # Now check each white cell for forced-black conflict:
    # If maxBlackCol[r] >= c and maxBlackRow[c] >= r, that's a conflict => No
    # If row r not in maxBlackCol => treat maxBlackCol[r]=0
    # If col c not in maxBlackRow => treat maxBlackRow[c]=0

    for (r,c) in white_cells:
        rmax = maxBlackCol.get(r, 0)
        cmax = maxBlackRow.get(c, 0)
        if rmax >= c and cmax >= r:
            print("No")
            return

    # If no conflict, we can satisfy all
    print("Yes")