def main():
    import sys
    data = sys.stdin.read().split()
    if not data: 
        return
    t = int(data[0])
    pos = 1
    out_lines = []
    # We explain the idea briefly:
    # In the problem we have an initial set of pieces at positions (where A has '1')
    # and a desired pattern of squares to “cover” (where B has '1'). Moves are unusual:
    # each operation moves every piece one step toward a chosen pivot.
    # One can show that in any sequence these moves move each piece by exactly one unit per move.
    # Therefore, if a piece initially at position p ends at position q,
    # then you must have used at least |p-q| moves.
    # In the final configuration, every square i with B[i]=='1' must receive at least one piece
    # and squares with B[i]=='0' are empty.
    #
    # Because pieces are indistinguishable and their relative order never changes,
    # one may “assign” each piece (in sorted order) a target position from Q – the indices where B == '1'.
    # (Some targets may get more than one piece, but every target must appear at least once.)
    # In an ideal plan the maximum distance any piece must travel (which is the minimal number of moves)
    # is exactly the smallest r for which there exists an assignment f from pieces to targets (non-decreasing,
    # and the set of targets used is exactly Q) satisfying |p - f(p)| <= r for every piece.
    #
    # Our plan is:
    #  1. Let P be the sorted list (1-indexed) of positions where A is '1' and Q be that for B.
    #  2. If len(P) < len(Q) it is impossible (since we cannot cover all target squares).
    #  3. Otherwise, binary search on r (from 0 to n) to find the minimum r for which it is possible to assign every piece
    #     a target from Q (in non-decreasing order) so that each piece moves at most r positions and every Q is chosen by at least one piece.
    #
    # For the check feasibility function (feas):
    # We simulate the following greedy assignment:
    #   - Let curr_group be the index (in Q) for the “current target value” (starting at 0).
    #   - For each piece (in sorted order from P), if it can “advance” the current assignment by covering the next target Q[curr_group+1]
    #     (i.e. if |piece - Q[curr_group+1]| <= r) then we do so – meaning that piece will serve as the first piece covering that target.
    #     Otherwise we assign it to Q[curr_group] (provided |piece - Q[curr_group]| <= r).
    # In the end we must have advanced (i.e. used) all targets.
    
    def feas(r, P, Q):
        # curr_group is our pointer into Q.
        curr_group = 0
        k = len(Q)
        for p in P:
            # If we haven't covered all targets yet, check if this piece can start covering the next target.
            if curr_group < k - 1 and abs(p - Q[curr_group + 1]) <= r:
                curr_group += 1
                # Once advanced, proceed to next piece.
                continue
            # Otherwise, this piece is “assigned” to the current target.
            # But if p is too far from the current group's target, then r is too small.
            if abs(p - Q[curr_group]) > r:
                return False
        # In a valid assignment every target must be used at least once.
        return curr_group == k - 1

    for _ in range(t):
        n = int(data[pos]); pos += 1
        A = data[pos]; pos += 1
        B = data[pos]; pos += 1

        P = []
        Q = []
        # Positions are 1-indexed.
        for i, ch in enumerate(A, start=1):
            if ch == '1':
                P.append(i)
        for i, ch in enumerate(B, start=1):
            if ch == '1':
                Q.append(i)

        m = len(P)
        k = len(Q)
        if m < k:
            out_lines.append("-1")
            continue
            
        lo = 0
        hi = n  # maximum possible displacement needed is at most n.
        ans = hi
        found = False
        while lo <= hi:
            mid = (lo + hi) // 2
            if feas(mid, P, Q):
                ans = mid
                found = True
                hi = mid - 1
            else:
                lo = mid + 1
        out_lines.append(str(ans) if found else "-1")
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()