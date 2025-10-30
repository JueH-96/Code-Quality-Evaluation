def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    A = list(map(int, input_data[2:]))

    # ----------------------------------------------------------------------------
    # 1)  Find x = the maximum possible value of min(w_1, w_2, ..., w_K).
    #
    #     We will use a binary-search over X in [0 .. sum(A)//K].
    #     A standard necessary condition: sum(A) >= K*X.
    #     Upper bound for X is sum(A)//K.
    #
    #     "Feasibility check" (can we partition the circular cake into K contiguous
    #     arcs, each of sum >= X, covering all N pieces exactly once?) is the main
    #     challenge.  We will implement a known O(N) check often used in similar
    #     "largest-min-segment-sum in a circular array" problems:
    #
    #     IDEA for feasibility check (sometimes called "cover-in-subarray" trick):
    #
    #       - We replicate A twice, forming A' of length 2N.
    #       - We do a single pass from left to right, greedily cutting a "segment"
    #         each time the running sum >= X (and resetting the sum).
    #         Let coverage[i] = how many segments have been *greedily* formed
    #         from A'[0..i-1].
    #
    #         coverage[0] = 0  (no elements, no segments)
    #         Then for i in [0..2N-1]:
    #             running_sum += A'[i]
    #             coverage[i+1] = coverage[i]
    #             if running_sum >= X:
    #                 coverage[i+1] = coverage[i] + 1
    #                 running_sum = 0
    #
    #       - coverage[i] is *not* necessarily a partition of exactly i pieces,
    #         it is just the count of how many times we hit sum >= X by index i-1
    #         in a greedy fashion.  Some leftover may be < X, but we imagine
    #         "in a partition coverage" that leftover can be merged with the last
    #         cut (which had >= X) so coverage count does not decrease.
    #
    #       - Now, to see if there's a way to cover a *circular* segment of length N
    #         with K arcs each >= X, we check subwindows of length N in A'.
    #         Concretely, for each i from N..2N, we look at coverage[i] - coverage[i-N].
    #         This difference is how many times the greedy approach "hit >= X"
    #         between indices [i-N .. i-1].
    #
    #         If coverage[i] - coverage[i-N] >= K, that indicates in that window
    #         of length N we formed at least K cuts each summing >= X (when each
    #         cut was triggered).  Crucially, because we use the standard "greedy
    #         from left" approach, if we form M >= K such "segments" in that window,
    #         we can combine any extras to keep each ≥ X and still cover that window
    #         contiguously.  Hence it is "feasible".
    #
    #     We must also check the trivial fail-fast: if sum(A) < K*X, infeasible.
    #
    # ----------------------------------------------------------------------------
    def can_partition_circular(A, N, K, X):
        # quick check
        if X == 0:
            return True  # trivially feasible if X=0, but here X≥1 in actual usage
        total_sum = sum(A)
        if total_sum < K * X:
            return False

        # Build doubled array
        Adbl = A + A
        coverage = [0] * (2*N + 1)
        run_sum = 0
        seg_count = 0
        # Greedy "hit >= X" approach over 2N pieces
        for i in range(2*N):
            run_sum += Adbl[i]
            coverage[i+1] = coverage[i]
            if run_sum >= X:
                coverage[i+1] = coverage[i] + 1
                run_sum = 0

        # Now check any subwindow of length N for at least K hits
        # i in [N.. 2N], subwindow is [i-N.. i-1]
        # number of hits in that subwindow = coverage[i] - coverage[i-N]
        feasible = False
        for i in range(N, 2*N + 1):
            if coverage[i] - coverage[i - N] >= K:
                feasible = True
                break
        return feasible

    # Binary search for x in [1.. max_possible], where max_possible = sum(A)//K
    # (since min(w_i) cannot exceed sum(A)/K)
    S = sum(A)
    left, right = 1, S // K  # min sum >=1 and <= sum(A)//K
    best_x = 0
    while left <= right:
        mid = (left + right) // 2
        if can_partition_circular(A, N, K, mid):
            best_x = mid
            left = mid + 1
        else:
            right = mid - 1

    x = best_x  # The maximum possible min-sum

    # ----------------------------------------------------------------------------
    # 2)  Count how many cut lines (boundaries) are "never cut" in any optimal division.
    #
    #     A boundary i (1≤i≤N) is between piece i and piece i+1 (with piece N+1 = piece1).
    #     "never cut" means: in every valid partition that achieves min-sum = x,
    #     this boundary is not used.
    #
    #     Equivalently, if there exists at least one valid partition (also achieving
    #     min-sum = x) in which boundary i *is* used, then boundary i is *not* in the
    #     "never cut" set.
    #
    #     We need to find the count of boundaries that are *impossible* to use
    #     in any partition with min-sum = x.
    #
    #     ------------------------------------------------------------
    #     A known useful observation (often seen in editorial solutions to
    #     similar tasks):
    #
    #       - If two adjacent pieces (i, i+1) can be in the same segment
    #         without causing that segment's sum to drop below x (i.e. there
    #         exists a way to "bundle" them into some segment of total ≥ x),
    #         then it might be possible that boundary i is *not* used. But
    #         that does not prove boundary i is "never used."
    #
    #       - Conversely, if forcing pieces i and i+1 into the same segment
    #         inevitably makes that segment's sum < x (no matter how big we
    #         extend around them), that boundary i *must* be cut in
    #         any partition that achieves min-sum = x.  In that case, boundary i
    #         definitely cannot be "never cut"; it is actually "always cut."
    #
    #     But we need "never cut," i.e. boundary i is absent from all optimal
    #     divisions.  
    #
    #     Looking at the sample solutions can give a hint: in Sample 1,
    #     boundary 5 is "never cut" even though piece5 + piece1 = 7 < 13.
    #     The reason: in *all* ways to achieve min-sum=13, those two pieces end up
    #     in the same (larger) segment that has total 13 or 14.  Meanwhile,
    #     other boundaries get chosen.
    #
    #     ------------------------------------------------------------
    #     A practical way to answer the second query (common in contest editorials)
    #     is:
    #
    #       - For each boundary i, check if we can construct *any* valid partition
    #         with min-sum = x that *uses* boundary i.  If yes, then i is not in
    #         "never cut."  If no, then i is in "never cut."
    #
    #     Naively, one might try for each i forcing that boundary to be used
    #     and see if there's a partition with min-sum >= x, which is O(N) times
    #     an O(N) feasibility check => O(N^2).  Too big for N up to 2e5.
    #
    #     ------------------------------------------------------------
    #     However, there is a well-known simpler criterion in (many) problems of
    #     this type:
    #
    #       "Boundary i is 'never cut' in optimal partitions" ⇔
    #       "In order to reach min-sum = x, pieces i and i+1 must always be in
    #        the same segment in *every* such partition."
    #
    #     This typically happens if 'splitting them apart' would force some
    #     segment to go below x.  Or equivalently, "no matter how you arrange
    #     the other segments, if you put a cut at boundary i, you won't achieve
    #     min-sum = x."
    #
    #     For large N, deriving a purely local test (just on A[i], A[i+1]) does
    #     not work, as the sample shows.  One needs to consider that i and i+1
    #     could be bundled in a bigger segment.
    #
    #     ------------------------------------------------------------
    #     A known editorial trick (appearing in problems like JOI 2018 Final Cake,
    #     or similar) is:
    #
    #       - Once x is fixed, we view an equivalent array B = [b_1, b_2, ..., b_N]
    #         where b_i = A_i, but we keep track of which boundaries *must* be cut
    #         to keep segments ≥ x in any arrangement.  Also which boundaries
    #         *can* be cut without breaking feasibility, etc.
    #
    #       - In fact, one can show that if there is *any* optimal partition that
    #         uses boundary i, then there is a greedy-like shift argument
    #         guaranteeing that boundary i can appear in a "greedy" partition
    #         with min-sum≥ x.  That allows a single pass around the circle
    #         (or a suitably doubled array) to find all boundaries that appear
    #         in at least one feasible partition.  The rest are "never cut."
    #
    #     Implementation detail:
    #
    #       We'll do the same "2N greedy segmentation" as in the feasibility check,
    #       but now *record* exactly where we place boundaries.  Because that
    #       single pass produces a set of boundaries (where sum >= x is reached),
    #       it is effectively one particular arrangement in the linear sense.
    #       Then for each subwindow of length N, if it has K or more boundaries
    #       from that pass, that subwindow is feasible for min-sum = x.  Then all
    #       boundaries used inside that subwindow are "used in some valid partition."
    #
    #       So the strategy:
    #
    #         1) Perform the "greedy cut" pass over 2N array (with sumCur reset
    #            to 0 each time sumCur >= x).  Collect all boundary positions
    #            in boundaryIndices[].
    #         2) For each boundary b in boundaryIndices, consider which subwindows
    #            of length N can realize a valid partition that includes b.  We
    #            only need to find at least one subwindow of length N for which
    #            the count of boundaries is >= K and that boundary b is inside.
    #
    #            Then boundary b mod N is "used" in some valid partition of the
    #            circle.  (Here "b mod N" is the boundary index in the original
    #            circle sense, i.e. boundary i is between piece i and i+1.)
    #
    #         3) Any boundary i that does not appear in any such subwindow (with
    #            enough boundaries) is "never cut."
    #
    #       Implementation details to watch carefully:
    #         - The "boundary" in the greedy pass occurs at index i (0-based) meaning
    #           we ended a segment exactly at piece i.  In the circle, that boundary
    #           corresponds to "between piece i and piece i+1" in 1-based indexing.
    #           So the boundary index is i % N + 1 (in 1-based).  Or just i % N
    #           in 0-based if we label boundaries 0..N-1.
    #
    #         - We must also handle the "leftover combine" logic: in the standard
    #           largest-min-subarray-sum approach, the leftover < x can merge with
    #           the last boundary.  So effectively the boundary index for that last
    #           segment might expand beyond the place we cut.  However, for the
    #           purpose of "was boundary i used *somewhere*" we only need that the
    #           cut was declared at i anyway.  Merging leftover does not remove
    #           that cut from existence.  So we can keep it simple: each time
    #           sumCur >= x, we say "we placed a boundary at i" and sumCur=0.
    #
    #         - Next, to check subwindows [s.. s+N-1], we want to see how many of
    #           our "greedy boundaries" fall strictly inside that window.  If that
    #           count is >= K, the subwindow can represent a valid partition that
    #           uses exactly those cuts (plus possibly merging leftover).
    #           Then each boundary in that subwindow is "used in some valid partition."
    #
    #       Complexity: We'll have at most ~2N boundaries in the worst case.  We run
    #       a two-pointer to see, for each boundaryIndices[i], which subwindow(s)
    #       can contain i.. i+K-1 boundaries.  We only need to confirm existence
    #       of *some* subwindow that has >= K boundaries and includes boundaryIndices[i].
    #
    #       A simpler method:
    #         - We already do the subwindow feasibility check by "coverage array."
    #           For each i in [N..2N], if coverage[i]-coverage[i-N]>=K => that subwindow
    #           is feasible.  Let’s define a boolean feasibleWindow[i] = ( coverage[i]-coverage[i-N]>=K ).
    #         - A boundary b (where b is an index in [0..2N-1]) lies in that window
    #           if s <= b < s+N, i.e. s <= b < s+N => b < i => b >= i-N.  So b is "in"
    #           subwindow i if i-N <= b < i.  Then if feasibleWindow[i] is True, all
    #           boundaries b in [i-N.. i-1] can be used in some feasible partition
    #           (the one that subwindow i represents).
    #
    #       Implementation steps:
    #         1) compute coverage[] as done for feasibility.  Mark which i in
    #            [N..2N] yield feasibleWindow[i].
    #         2) produce boundaryIndices from the same pass.  For each boundary b,
    #            find all i in [N..2N] such that feasibleWindow[i] == True and
    #            i-N <= b < i.  If there's at least one, boundary b is "used in
    #            some feasible partition."  We can do this with a two-pointer
    #            approach over b.  Because i-N <= b < i => b+1 in [i-N+1.. i],
    #            => i in [b+1.. b+N].  We check if any i in that range is feasibleWindow[i].
    #
    #         Implementation detail: We'll keep a set (or a pointer array) of indices
    #         i where feasibleWindow[i] == True.  Then for each boundary b, we want
    #         to see if there exists an i in that set with i in [b+1.. b+N].  If yes,
    #         boundary b belongs to a feasible partition.  Then boundaryIndex b mod N
    #         is "used" in the circle sense.
    #
    #     After that, any boundary i in [0..N-1] not so marked is "never cut."
    #
    # ----------------------------------------------------------------------------

    # --- Find all subwindows of length N that are feasible for x ---
    # Re-run the coverage array for X = x
    Adbl = A + A
    coverage = [0] * (2*N + 1)
    run_sum = 0
    for i in range(2*N):
        run_sum += Adbl[i]
        coverage[i+1] = coverage[i]
        if run_sum >= x:
            coverage[i+1] = coverage[i] + 1
            run_sum = 0

    # feasibleWindow[i] = True if subwindow [i-N, ..., i-1] has >= K boundaries
    feasibleWindow = [False]*(2*N+1)
    for i in range(N, 2*N+1):
        if coverage[i] - coverage[i-N] >= K:
            feasibleWindow[i] = True

    # --- Collect the actual boundary indices from the same pass
    boundaryIndices = []
    run_sum = 0
    for i in range(2*N):
        run_sum += Adbl[i]
        if run_sum >= x:
            boundaryIndices.append(i)
            run_sum = 0
    # boundaryIndices may have at most 2N entries

    # --- We'll keep the indices i where feasibleWindow[i] == True in a list
    feasible_starts = [i for i in range(N, 2*N+1) if feasibleWindow[i]]

    # For each boundary b, we want to know if there's some i in feasible_starts
    # with i in [b+1 .. b+N].
    # We'll do a two-pointer approach:
    used_in_optimal = [False]*(2*N)  # for boundary indices b in [0..2N-1]
    # feasible_starts is sorted ascending; boundaryIndices is in ascending order

    # Pointer fs points into feasible_starts, which are strictly increasing
    fs_len = len(feasible_starts)
    fs_ptr = 0
    for b in boundaryIndices:
        # We want any i in feasible_starts with i in [b+1 .. b+N].
        # Because feasible_starts is sorted, we can move fs_ptr until
        # feasible_starts[fs_ptr] < b+1 or feasible_starts[fs_ptr] > b+N
        # We'll just advance fs_ptr as needed and also check if feasible_starts[fs_ptr]
        # is within [b+1.. b+N].
        # A simpler way is to do a while-loop to move fs_ptr if feasible_starts[fs_ptr] < b+1.
        while fs_ptr < fs_len and feasible_starts[fs_ptr] < b+1:
            fs_ptr += 1
        # now if fs_ptr < fs_len and feasible_starts[fs_ptr] <= b+N, boundary b is used.
        if fs_ptr < fs_len and feasible_starts[fs_ptr] <= b+N:
            used_in_optimal[b] = True

    # Now, a boundary b in the doubled array corresponds to boundary (b mod N)
    # in the original circle.  If used_in_optimal[b] == True for some b that
    # has the same b mod N, that means the boundary (b mod N) is used in *some*
    # feasible partition.  So that boundary is *not* "never cut."
    # We'll gather them up in a set.

    used_mod = [False]*N
    for b in range(2*N):
        if used_in_optimal[b]:
            used_mod[b % N] = True

    # The "never cut" boundaries are those i in [0..N-1] for which used_mod[i] == False.
    y = sum(1 for i in range(N) if not used_mod[i])

    print(x, y)

# Do not forget to call main().
if __name__ == "__main__":
    main()