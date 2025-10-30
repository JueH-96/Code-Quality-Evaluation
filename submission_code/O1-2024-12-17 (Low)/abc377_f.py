def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])

    # Read in the positions of the M pieces
    pieces = [(int(input_data[2+2*k]), int(input_data[3+2*k])) for k in range(M)]

    #--------------------------------------------------------------------------------
    # We have an N x N board (N up to 10^9), and M pieces (M up to 10^3).
    # Each piece at (r, c) threatens:
    #   - all squares in row r
    #   - all squares in column c
    #   - all squares on the diagonal i+j = r+c
    #   - all squares on the anti-diagonal i-j = r-c
    #
    # We want to count how many empty squares (i.e., not occupied already) remain
    # that are NOT threatened by any of the M pieces.
    #
    # Let us define four sets (by their "labels"):
    #   R = { all row-indices that have a piece }
    #   C = { all column-indices that have a piece }
    #   S1 = { all values d = r+c for a piece at (r,c) }
    #   S2 = { all values e = r-c for a piece at (r,c) }
    #
    # A square (i,c) is threatened if:
    #   i in R, or
    #   c in C, or
    #   i + c in S1, or
    #   i - c in S2.
    #
    # However, some squares are already occupied by the M existing pieces. Those are
    # obviously not available to place our new piece either. So at the very end we must
    # also subtract off those M squares from the final count if they were not
    # already excluded by the threat sets.  But in fact, for the answer
    # "how many empty squares can I safely place on?" we can do:
    #
    #   safe_squares = N^2 - (number_threatened_by_any_piece) - (M squares already occupied)
    #
    # But we must be careful not to subtract the M occupied squares twice in case
    # they also appear among threatened squares. The correct approach is:
    #   total_empty = N^2 - M  (because M squares are occupied)
    #   threatened_union = (threatened by rows) union (columns) union (diag+) union (diag-)
    #   But we do not want to count the squares that are in the union but are also among
    #   the M occupied squares as "subtracting from empty" more than once.
    #
    # Actually it can be simpler to do an Inclusion-Exclusion for the union of threatened sets:
    #   Let A = set of squares threatened by any row in R. (|A| = |R|*N)
    #   Let B = set of squares threatened by any column in C. (|B| = |C|*N)
    #   Let C_ = set of squares threatened by diagonals i+j in S1.
    #   Let D_ = set of squares threatened by anti-diagonals i-c in S2.
    # Then |A∪B∪C_∪D_| = sum of single counts
    #                     - sum of pairwise intersections
    #                     + sum of triple intersections
    #                     - sum of quadruple intersection
    #
    # Finally, the squares where we could place a new piece is:
    #   (N^2 - M) - (|A∪B∪C_∪D_| - overlap_with_occupied)
    #
    # but we have to be mindful about how the M occupied squares factor in. In effect,
    # those M squares are already "not available," so the total "empty" is N^2 - M.
    # Next, from those empty squares, remove those that are threatened. But if an
    # occupied square is also threatened, it does not matter, because it's already
    # not in the empty set. So we want:
    #
    #   safe_count = (N^2 - M) - ( count of threatened squares among the empty ones )
    #
    # The threatened squares among the empty ones is simply all threatened squares
    # minus the ones that are actually occupied. So effectively:
    #
    #   threatened_among_empty = |A∪B∪C_∪D_| - |(A∪B∪C_∪D_) ∩ (occupied)|.
    #
    # The set of occupied squares is precisely those M distinct squares from input.
    # We can count how many of those M squares fall into the threatened union by
    # simply checking each occupied square (r,c) if it belongs to A or B or C_ or D_.
    # If it is threatened, that means it was counted in the union. We'll count how many
    # occupied squares are threatened; call that T_occ.
    #
    # Then:
    #   safe_count = (N^2 - M) - (|A∪B∪C_∪D_| - T_occ).
    #              = N^2 - |A∪B∪C_∪D_| - M + T_occ
    #
    # We'll do an inclusion-exclusion for |A∪B∪C_∪D_|:
    #
    #   Let:
    #     A_count = |A| = |R| * N
    #     B_count = |B| = |C| * N
    #     C_count = sum of line lengths for each d in S1
    #     D_count = sum of line lengths for each e in S2
    #
    #   Pairwise intersections:
    #     A∩B: squares threatened by row i in R and col j in C => exactly |R|*|C|.
    #     A∩C: squares threatened by row i in R and i+j in S1 => for each i in R, for each d in S1,
    #            we get the single col j = d - i, if 1 <= j <= N. Count how many are valid.
    #     A∩D: similarly for row i in R, i-j in S2 => j = i - e. Count valid j in [1..N].
    #     B∩C: for col j in C, i+j in S1 => i = d - j. Count valid i in [1..N].
    #     B∩D: for col j in C, i-j in S2 => i = j + e. Count valid i in [1..N].
    #     C∩D: lines i+j = d, i-j = e intersect in at most one square i=(d+e)/2, j=(d-e)/2,
    #           if those are integers in [1..N]. We'll loop over all d in S1,
    #           e in S2, check parity and bounds. Count.
    #
    #   Triple intersections:
    #     A∩B∩C: squares in row i in R, col j in C, i+j in S1 => just check i+j in S1 for each i in R, j in C.
    #     A∩B∩D: squares in row i in R, col j in C, i-j in S2 => check i-j in S2 for each i in R, j in C.
    #     A∩C∩D: squares in row i in R, i+j in S1, i-j in S2 => for each (d,e) in S1 x S2, i=(d+e)//2,
    #                check i in R and c=(d-e)//2 in [1..N]. Must also check parity, bounds, etc.
    #     B∩C∩D: squares in col j in C, i+j in S1, i-j in S2 => for each (d,e) in S1 x S2, j=? We solve:
    #                i=(d+e)//2, j=(d-e)//2, then check j in C, etc.
    #
    #   Quadruple intersection A∩B∩C∩D: row i in R, col j in C, i+j in S1, i-j in S2.
    #     We can do i in R, j in C, check if i+j in S1 and i-j in S2.
    #
    # This is all feasible for M <= 1000, because each set has size at most 1000,
    # and the intersection calculations each take at most O(1000^2) = 10^6 steps,
    # which is borderline but doable in optimized Python if we are careful.
    #
    # Then we compute T_occ = how many of the M occupied squares lie in the threatened union.
    #   A square (r,c) is threatened if r in R or c in C or r+c in S1 or r-c in S2.
    #   We'll check each of the M squares in O(1) set membership. Sum up how many are threatened.
    #
    # Finally:
    #
    #   threatened_union = inclusion_exclusion_value
    #   T_occ = count of occupied squares in that union
    #   safe_count = N^2 - threatened_union - M + T_occ
    #
    # Print safe_count.
    #
    # NOTE: watch out for 64-bit / big integer usage. Python can handle big integers,
    # but we must be sure to keep track carefully.
    #
    # Implementation steps now:
    #--------------------------------------------------------------------------------

    # 1) Build sets of threatened rows, columns, diag+, diag-:
    Rset = set()
    Cset = set()
    S1set = set()  # i+j
    S2set = set()  # i-j

    occupied = set()  # store the occupied squares for final check

    for (r, c) in pieces:
        Rset.add(r)
        Cset.add(c)
        S1set.add(r + c)
        S2set.add(r - c)
        occupied.add((r, c))

    R = sorted(Rset)
    C = sorted(Cset)
    S1 = sorted(S1set)
    S2 = sorted(S2set)

    # A_count = |R| * N
    A_count = len(R) * (N)

    # B_count = |C| * N
    B_count = len(C) * (N)

    # C_count = sum_{d in S1} line_length_plus(d, N)
    # line_length_plus(d,N):
    #   if d < 2 or d > 2N => 0
    #   if 2 <= d <= N+1 => d-1
    #   if N+1 < d <= 2N => 2N - d + 1
    def line_length_plus(d, N):
        # i+j = d
        # valid i in [1..N], j in [1..N]
        # => the line length is:
        if d < 2 or d > 2*N:
            return 0
        if d <= N+1:
            return d - 1
        else:
            return 2*N - d + 1

    C_count = 0
    for d in S1:
        C_count += line_length_plus(d, N)

    # D_count = sum_{e in S2} line_length_minus(e, N)
    # line_length_minus(e,N):
    #   squares with i-j = e
    #   => i = j + e
    #   j in [1..N], i in [1..N]
    #   => j+e in [1..N] => j in [1-e..N-e]
    #   => intersection with [1..N] => length = N - |e| (if -N+1 <= e <= N-1), else 0
    def line_length_minus(e, N):
        # i-j = e
        # valid if 1 <= j <= N, and 1 <= j+e <= N
        # number of j that satisfy max(1,1-e) <= j <= min(N, N-e)
        # length = N - |e| if e in [-(N-1)..(N-1)], else 0
        if e < - (N - 1) or e > (N - 1):
            return 0
        return N - abs(e)

    D_count = 0
    for e in S2:
        D_count += line_length_minus(e, N)

    # Now pairwise intersections:

    # A ∩ B => row in R, col in C => each pair is a distinct square => |R|*|C|
    AB_count = len(R) * len(C)

    # A ∩ C => row i in R, i+j = d in S1 => j = d - i in [1..N]
    # We'll do a nested loop ( up to 1000*1000 => 1e6 ), check validity
    AC_count = 0
    Rset_lookup = R  # already have it, but we'll just loop
    S1set_lookup = S1
    import bisect

    # We'll just do straightforward approach:
    # for i in R:
    #   for d in S1:
    #       j = d - i
    #       if 1 <= j <= N => AC_count++
    AC_count_local = 0
    idx_r = 0
    idx_s1 = 0
    # Straight nested:
    for i in R:
        for d in S1:
            j = d - i
            if 1 <= j <= N:
                AC_count_local += 1
    AC_count = AC_count_local

    # A ∩ D => row i in R, i-j = e in S2 => j = i - e in [1..N]
    AD_count_local = 0
    for i in R:
        for e in S2:
            j = i - e
            if 1 <= j <= N:
                AD_count_local += 1
    AD_count = AD_count_local

    # B ∩ C => col j in C, i+j=d in S1 => i=d-j => i in [1..N]
    BC_count_local = 0
    for j in C:
        for d in S1:
            i = d - j
            if 1 <= i <= N:
                BC_count_local += 1
    BC_count = BC_count_local

    # B ∩ D => col j in C, i-j=e in S2 => i=e+j => i in [1..N]
    BD_count_local = 0
    for j in C:
        for e in S2:
            i = j + e
            if 1 <= i <= N:
                BD_count_local += 1
    BD_count = BD_count_local

    # C ∩ D => lines i+j = d, i-j = e => at most 1 square if (d+e) even => i=(d+e)//2, j=(d-e)//2
    CD_count_local = 0
    for d in S1:
        for e in S2:
            # check parity
            if ((d + e) & 1) == 0:
                i = (d + e) >> 1
                j = (d - e) >> 1
                if 1 <= i <= N and 1 <= j <= N:
                    CD_count_local += 1
    CD_count = CD_count_local

    # Triple intersections:

    # A ∩ B ∩ C => row i in R, col j in C, i+j in S1
    ABC_count_local = 0
    for i in R:
        for j in C:
            if (i + j) in S1set:
                ABC_count_local += 1
    ABC_count = ABC_count_local

    # A ∩ B ∩ D => row i in R, col j in C, i-j in S2
    ABD_count_local = 0
    for i in R:
        for j in C:
            if (i - j) in S2set:
                ABD_count_local += 1
    ABD_count = ABD_count_local

    # A ∩ C ∩ D => row i in R, i+j in S1, i-j in S2 => intersection of lines with a fixed i
    # but we do it by scanning (d,e) in S1 x S2 => i=(d+e)//2, j=(d-e)//2 => check if i in R
    # (since that i must be row in R), and 1<=i,j<=N
    ACD_count_local = 0
    Rhash = set(R)  # for O(1) membership
    for d in S1:
        for e in S2:
            if ((d + e) & 1) == 0:
                i = (d + e) >> 1
                j = (d - e) >> 1
                if 1 <= i <= N and 1 <= j <= N:
                    if i in Rhash:
                        ACD_count_local += 1
    ACD_count = ACD_count_local

    # B ∩ C ∩ D => col j in C, i+j in S1, i-j in S2 => i=(d+e)//2, j=(d-e)//2
    # we want j in C => do the same approach but check j in C
    BCD_count_local = 0
    Chash = set(C)
    for d in S1:
        for e in S2:
            if ((d + e) & 1) == 0:
                i = (d + e) >> 1
                j = (d - e) >> 1
                if 1 <= i <= N and 1 <= j <= N:
                    if j in Chash:
                        BCD_count_local += 1
    BCD_count = BCD_count_local

    # Quadruple intersection: A ∩ B ∩ C ∩ D => row i in R, col j in C, i+j in S1, i-j in S2
    # we can do i in R, j in C => check i+j in S1, i-j in S2
    ABCD_count_local = 0
    for i in R:
        for j in C:
            if (i + j) in S1set and (i - j) in S2set:
                ABCD_count_local += 1
    ABCD_count = ABCD_count_local

    # Now compute |A ∪ B ∪ C_ ∪ D_|
    # by inclusion-exclusion:
    #   union_count = A + B + C + D
    #                - (AB + AC + AD + BC + BD + CD)
    #                + (ABC + ABD + ACD + BCD)
    #                - (ABCD)
    union_count = (
        A_count + B_count + C_count + D_count
        - (AB_count + AC_count + AD_count + BC_count + BD_count + CD_count)
        + (ABC_count + ABD_count + ACD_count + BCD_count)
        - (ABCD_count)
    )

    # Next T_occ = how many of the M occupied squares (r,c) are threatened
    # i.e. r in R or c in C or r+c in S1 or r-c in S2
    # We'll just check membership in O(1) sets for each square
    threat_occupied = 0
    for (r, c) in occupied:
        if (r in Rset) or (c in Cset) or ((r + c) in S1set) or ((r - c) in S2set):
            threat_occupied += 1

    # The number of safe squares is:
    #   safe_count = N^2 - union_count - M + threat_occupied
    # because union_count counts all threatened squares (including some that might be
    # occupied). Among those union_count squares, threat_occupied of them are occupied.
    # So the threatened squares that are actually empty is union_count - threat_occupied.
    # Hence from the (N^2 - M) empty squares, we remove (union_count - threat_occupied).
    #
    # => safe_count = (N^2 - M) - (union_count - threat_occupied)
    #               = N^2 - union_count - M + threat_occupied
    #
    # Make sure it doesn't go negative; it shouldn't in principle, but we can clamp to 0
    # in case.  (The problem statement doesn't imply a scenario that would yield negative.)
    #
    safe_count = (N*N) - union_count - M + threat_occupied
    if safe_count < 0:
        safe_count = 0  # safeguard

    print(safe_count)


# Do not forget to call main()
if __name__ == "__main__":
    main()