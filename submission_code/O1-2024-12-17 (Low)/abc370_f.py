def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    A = list(map(int, input_data[2:]))

    # -------------------------------------------------------------------------
    # 1) We want to find x = the maximum possible value of min(w_1, ..., w_K).
    #
    #    Each of K people gets one (circularly) consecutive nonempty block
    #    of pieces, covering all N pieces exactly once. We want to maximize
    #    the smallest block-sum.
    #
    #    A standard technique is binary-search on x. We test "feasible(x)" by
    #    seeing if we can form K segments on the circle, each with sum >= x.
    #
    #    For the CIRCULAR array, a known trick is to "double" the array into
    #    B = A + A (length 2N), place "cut boundaries" greedily wherever a
    #    running sum >= x, and then check if we can pick a set of K boundaries
    #    that fit in some consecutive block of length N. If so, it's feasible.
    #
    # 2) After finding x, we need y = the count of cut lines that are NEVER used
    #    in ANY valid partition having min-block-sum = x. A "cut line i" is the
    #    boundary between piece i and piece i+1 (circularly, piece N+1 = piece 1).
    #
    #    The approach here is:
    #      - Having x fixed, we again do the same "greedy boundary" pass over
    #        B = A + A. We record all possible boundaries where a segment of
    #        sum >= x can be closed.
    #      - Then, for each consecutive set of K boundaries in that list that
    #        fit within a window of length N, that set represents a valid K-part
    #        in the circle (from some start index to start+N-1). We mark all
    #        boundaries in that chosen set as "used in at least one valid
    #        partition."
    #
    #    In the end, any boundary not marked is "never used" in any solution,
    #    so we count them.
    #
    #    Subtlety: Because we do a greedy pass that puts a boundary as soon as
    #    sum >= x, we actually produce MORE boundaries than necessary for a K-block
    #    partition. Indeed, from that "boundary-rich" list, we only need to choose
    #    K of them that fit in a window of length N. If it is possible to form
    #    that partition at all, those boundaries must come from this "boundary-rich"
    #    set. Hence if a boundary can appear in some valid partition, it must appear
    #    in the set of boundaries that the greedy pass flags. Therefore, marking
    #    all boundaries that can appear in any valid K-subset inside some length-N
    #    window from that set suffices to find which lines are "ever used."
    #
    # Complexity:
    #    - Summation of A_i up to ~2e9, so x up to ~1e9
    #    - We'll do ~log(1e9) ~ 30 steps of feasibility checks.
    #    - Each feasibility check does one pass over 2N (which is up to 400k),
    #      plus a sliding-window approach to see if we can pick K boundaries in
    #      some length-N window. This should be around 30 * 400k ~ 12 million
    #      operations, borderline but should be doable in optimized Python/PyPy.
    #
    # We'll implement carefully.
    # -------------------------------------------------------------------------

    # Prefix sums for A to speed total sum and to set upper bound for binary search
    total_sum = sum(A)
    # The maximum possible min-block-sum is at most total_sum//K.
    # We'll binary-search in [0..(total_sum//K)+1]

    # We'll build a prefix sums array for B = A + A to quickly compute subarray sums.
    # B length = 2N
    B = A + A
    prefix = [0] * (2*N+1)
    for i in range(2*N):
        prefix[i+1] = prefix[i] + B[i]

    def can_form_k_in_window(k, window_start, window_length, boundaries):
        """
        Given a sorted list of boundary indices (boundaries),
        check if there's a consecutive subset of size k whose first
        boundary >= window_start and last boundary <= window_start + window_length - 1.
        We only need to find if ANY subset of size k fits.

        Instead of an explicit combinatorial approach, we do a sliding window
        of size k over 'boundaries' to see if the difference boundaries[i+k-1] - boundaries[i] < window_length.
        But we also must ensure that boundaries[i] >= window_start.

        We'll do a binary search to find the range of boundaries that lie within
        [window_start, window_start + window_length - 1], then see if we can pick
        k consecutive boundaries inside that range.
        """
        import bisect
        left_idx = bisect.bisect_left(boundaries, window_start)
        right_idx = bisect.bisect_right(boundaries, window_start + window_length - 1) - 1
        # We want to see if right_idx - left_idx + 1 >= k, i.e. we have at least k boundaries in that range
        if right_idx - left_idx + 1 < k:
            return False

        # Actually we just need them to be consecutive. So if the # of boundaries in that range >= k, yes we can pick k.
        # But we must ensure that the sum of each segment is >= x, which was how we built 'boundaries' in the first place.
        return True

    def feasible(x):
        """
        Return True if we can split the circle into K segments
        each with sum >= x, else False.

        Implementation:
         - We'll do a "greedy pass" over B to find all boundary indices b_i
           (0 <= b_i < 2N) where a segment can be closed with sum >= x.
           Precisely: we start sumSeg=0 from index 0 of B. We move forward until sumSeg >= x,
           then place a boundary at that index. Then sumSeg=0 and continue. We'll gather
           all such boundary indices in ascending order. We do it in one sweep from i=0..2N-1.

           Then, for each window_start in [0..N-1], we see if we can pick K boundaries
           all within [window_start..window_start+N-1]. If yes, then it's feasible.

         - We'll do a more direct approach: as soon as we find a boundary in the range
           [start..start+N-1], that's 1 segment; we continue until we get K segments
           or surpass start+N-1.

         But to avoid O(N^2), we do:

         1) We build 'boundaries' by a single pass from left to right in B.
            That might produce up to about sum(B)/x boundaries, but in worst case ~2N if A_i are large or small.
         2) Then we apply a standard "sliding window of size N" technique: for each start in [0..N-1],
            we check if we can find K boundaries in [start..start+N-1]. We do this by
            scanning 'boundaries' with a pointer. We'll maintain a pointer that moves forward
            only, so we handle all starts in total O(len(boundaries)) time.
        """
        # 1) Build the "boundary candidates"
        boundaries = []
        ssum = 0
        start_idx = 0
        for i in range(2*N):
            ssum += B[i]
            if ssum >= x:
                boundaries.append(i)
                ssum = 0
        # boundaries is sorted in ascending order by construction.

        # 2) We want to see if there's ANY start in [0..N-1] such that we can pick
        #    at least K boundaries in [start..start+N-1].
        # We'll do a sliding approach with two pointers over the 'boundaries' array.
        # Specifically, let p=0.. len(boundaries)-1 be the left boundary in 'boundaries',
        # and we want to see if there's an index p+k-1 also in 'boundaries' with
        # boundaries[p+k-1] <= start+N-1. Then we vary 'start'.

        # We'll maintain a queue of how many boundaries lie in the interval [start..start+N-1].
        # As start goes from 0..N-1, the interval slides by 1. We'll track how many boundaries
        # fall out from the left, and how many come in from the right. Then we check if
        # the count >= K at any time.
        #
        # Implementation detail: We'll keep two pointers: leftP and rightP over 'boundaries'.
        # We'll move rightP forward while boundaries[rightP] < start + N. Then #in-window = rightP - leftP
        # We'll move leftP forward while boundaries[leftP] < start. We see if #in-window >= K.
        #
        # If for any start we get #in-window >= K, feasible -> True. If no start works, -> False.

        if not boundaries:
            return False

        leftP = 0
        rightP = 0
        count_in_window = 0

        feasible_found = False

        # We'll sweep start from 0..N-1 in ascending order.
        # For each start, the valid boundary range is [start .. start+N-1].
        # We'll move rightP forward while boundaries[rightP] <= start+N-1
        # We'll move leftP forward while boundaries[leftP] < start
        # count_in_window = rightP - leftP
        # check if count_in_window >= K

        # We do it in a single pass if we imagine start also increments by 1 each step.
        # We'll process them in sorted order of 'start'.
        # We'll keep adjusting leftP and rightP so that they correspond to the new interval.

        for start in range(N):
            # remove from leftP while boundaries[leftP] < start
            while leftP < len(boundaries) and boundaries[leftP] < start:
                leftP += 1
            # move rightP while boundaries[rightP] <= start+N-1
            while rightP < len(boundaries) and boundaries[rightP] <= start + N - 1:
                rightP += 1
            count_in_window = rightP - leftP
            if count_in_window >= K:
                feasible_found = True
                break

        return feasible_found

    # ---- Binary Search for the maximum x ----
    low, high = 0, (total_sum // K) + 1
    while low + 1 < high:
        mid = (low + high) // 2
        if feasible(mid):
            low = mid
        else:
            high = mid
    x = low  # maximum feasible min-block-sum

    # -------------------------------------------------------------------------
    # Now we must count how many cut lines are NEVER used in ANY valid partition
    # with min-block-sum = x.
    #
    # We'll repeat the "boundary building" with x, which gets us a set of "candidate"
    # boundaries in the doubled array B. Then from those boundaries, we check all
    # sliding windows of length N to see if we can pick K boundaries. If we can, we
    # mark those K boundaries as "used" in at least one valid partition. However,
    # note we have more than K boundaries in the range. We only need to pick ANY K
    # that form a valid partition. But from that set, effectively ANY boundary
    # that is inside that [start..start+N-1] window is "potentially used" in some
    # K-subset. Because we only need K of them, and if there are M >= K, we can pick
    # any K among them. So in fact, if a boundary B_i is inside the window, it
    # could be chosen. Therefore, whenever we see #in-window >= K, we can mark all
    # boundaries in that window as "used". Because some subset of them leads to
    # a valid partition. 
    #
    # Then, we reduce mod N to find which of the real cut lines are used. 
    #
    # Implementation detail:
    #   We'll build "boundaries_x" again. Then the same sliding approach from start=0..N-1:
    #   if #in-window >= K, we mark all those boundaries in [start..start+N-1] as used.
    #   In the end, any boundary i in [0..N-1] not marked is never used, so we count them.
    # -------------------------------------------------------------------------

    # Build boundaries_x
    boundaries_x = []
    ssum = 0
    start_idx = 0
    for i in range(2*N):
        ssum += B[i]
        if ssum >= x:
            boundaries_x.append(i)
            ssum = 0

    used = [False]*(N)  # used[i] => boundary i is used ( cut between piece i and i+1 ), 0-based

    if boundaries_x:  # if empty, then x=0 was maybe the only feasible, but let's handle normally
        leftP = 0
        rightP = 0
        for start in range(N):
            while leftP < len(boundaries_x) and boundaries_x[leftP] < start:
                leftP += 1
            while rightP < len(boundaries_x) and boundaries_x[rightP] <= start+N-1:
                rightP += 1
            count_in_window = rightP - leftP
            if count_in_window >= K:
                # mark all boundaries_x[p] for p in [leftP..rightP-1] as used mod N
                # because any of them could appear in a K-subset. 
                for p in range(leftP, rightP):
                    bidx = boundaries_x[p] % N
                    used[bidx] = True

    # count how many are never used
    never_used_count = sum(not u for u in used)

    # Output
    print(x, never_used_count)

# Do not forget to call main()
if __name__ == "__main__":
    main()