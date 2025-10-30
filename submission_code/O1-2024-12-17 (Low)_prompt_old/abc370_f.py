def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    A = list(map(int, input_data[2:]))

    # ------------------------------------------------------------------
    # 1) We want to find the largest possible value X = min(w_1, ..., w_K),
    #    where the cake (circular array of N pieces) is divided into K
    #    contiguous "arcs" (in circular sense), and w_i is the sum of
    #    masses in the i-th arc.
    #
    # 2) We also want to find how many of the cut lines (boundaries
    #    between piece i and piece i+1) are NEVER used in ANY optimal
    #    solution (i.e. all solutions that achieve that maximum X).
    #
    # ------------------------------------------------------------------
    # OVERVIEW OF APPROACH
    #
    # Let S = sum(A). Clearly, if we want each of the K arcs to have sum
    # at least X, then K*X <= S. Hence X <= floor(S/K). So we can do a
    # binary search for X in [0 .. floor(S/K)] to find the maximum X
    # such that we can partition the circle into K arcs each of sum >= X.
    #
    # CHECKING FEASIBILITY ("can we form K arcs with sum >= x?"):
    #
    #   Because the array is circular, a standard trick is to consider
    #   the array A "doubled": B = A + A (length 2N). A valid circular
    #   K-part partition of A corresponds to choosing some contiguous
    #   block of length N within B and dividing it into K contiguous
    #   arcs, each with sum >= x.  We only need to know if there EXISTS
    #   such a block/partition for some rotation. 
    #
    #   The classical approach is:
    #   - We do a greedy pass through B, forming arcs that each have sum >= x
    #     as soon as possible (i.e. minimal segmentation if each arc
    #     requires sum >= x). We collect the boundary indices in an array.
    #   - Then we check if there exist K consecutive arcs among that minimal
    #     segmentation that fit entirely in a window of length N.
    #
    #   With sum >= x, the "minimal segmentation" approach is:
    #     Initialize sum_curr = 0, boundary_list = [0]
    #     for i from 1..2N:
    #       sum_curr += B[i-1]
    #       if sum_curr >= x:
    #          boundary_list.append(i)
    #          sum_curr = 0
    #     Then boundary_list holds the ends of each arc in that greedy partition.
    #
    #   Suppose boundary_list = b0, b1, ..., bM   (b0 = 0)
    #   The arcs are [b0..b1-1], [b1..b2-1], ...
    #
    #   Now we want to see if there exists an index j so that
    #       b_{j+K} - b_j <= N
    #   and also we cover exactly N pieces.  The condition "b_{j+K} - b_j >= N"
    #   or "== N" can appear in some variants.  In practice, to cover exactly N
    #   consecutive pieces in B, we want the subarray from b_j (inclusive) to
    #   b_j + N - 1 (inclusive).  If b_{j+K} <= b_j + N, that means K arcs
    #   fit within that block of length <= N.  For them actually to cover
    #   exactly N pieces, we want b_{j+K} >= b_j + N as well (i.e. the last
    #   boundary is at or beyond b_j + N).  So we want:
    #
    #        b_j + N <= b_{j+K}   (so that the K arcs can span at least N pieces)
    #   and
    #        b_{j+K} - b_j <= N   (so that they do not exceed that block by more than N).
    #
    #   Putting it together typically implies b_{j+K} - b_j == N.  We check if
    #   that equality can appear among consecutive boundaries j..j+K in the
    #   minimal segmentation.  If yes, we say "feasible".
    #
    # In practice, a simpler robust check is often:
    #    We'll build the minimal arcs from left to right in B.  Collect boundary indices in bList.
    #    Then, for each j from 0..(len(bList)-K), check if bList[j+K] - bList[j] == N.
    #    If we ever see that, we say "feasible".
    #
    # That check is O(N) if the partition has at most O(N) arcs. Doing it for each binary
    # search step is O(N log(maxA)) which is acceptable for N=2e5 in optimized code
    # with a fast language. In Python, we'll need efficient I/O and careful coding,
    # but it can be done.
    #
    # Once we find the maximum X, next we need to count the number of cut lines that
    # are NEVER used in any optimal solution.  We'll approach that by:
    #
    #   (a) Find X using the above feasibility check.
    #   (b) Find all ways (or all boundary positions) that can appear in *some*
    #       valid partition that achieves X.  Then any boundary that never appears
    #       is "never cut."  We want the number of such boundaries.
    #
    #   For step (b), we can do the following:
    #   - We'll do the "minimal segmentation" in B for the value X. This yields
    #     boundary_list_min. Minimal segmentation tries to cut as soon as sum >= X,
    #     which yields the maximum number of arcs. But we are interested in *any*
    #     valid partition with K arcs. 
    #   - However, there's a known property: For a given X, if an index can appear
    #     as a boundary in *any* valid K-arc solution, then it must appear as a
    #     boundary in some "maximal or minimal segmentation" approach. The details
    #     are subtle, but a well-known fact is that to check the possible boundaries,
    #     one can also consider the "reverse minimal segmentation": scanning from
    #     right to left, or a "maximal segmentation" from left to right by waiting
    #     as long as you can until you'd violate the sum >= X for the next piece.
    #     Then the union of boundaries from these two greedy extremes often yields
    #     all possible boundaries that can appear in some valid segmentation.
    #
    #   - So, we can run two greeds:
    #        forward_min_greedy (cut as soon as sum >= X)
    #        forward_max_greedy (cut as late as possible but ensuring each segment >= X)
    #     done for the 2N array, gather boundary indices that can help achieve
    #     exactly N consecutive pieces with K arcs, if possible. Then also
    #     consider rotations. We'll gather all boundary indices that appear in
    #     either approach for any suitable rotation that leads to a valid covering
    #     of length N with K segments. Then, mapping those boundaries back to the
    #     circle, any boundary not in that set is "never cut" in any solution.
    #
    # Implementation details can get quite involved. We provide a reasonably
    # careful implementation that:
    #   1) finds X by binary search + feasibility.
    #   2) gathers all possible boundaries used in solutions that achieve X
    #      by running both "forward_min_greedy" and "forward_max_greedy," scanning
    #      all arcs in B, seeking subarrays of length N that yield K segments,
    #      and collecting the boundaries that matter. 
    #   3) The answer y is N minus the size of the set of used boundaries.
    #
    # ------------------------------------------------------------------
    # DUE TO THE COMPLEXITY, we implement carefully but succinctly.
    #
    # We'll define "build_boundaries_min" and "build_boundaries_max" which each
    # return a boundary list in [0..2N]. Then to check feasibility for x, we'll
    # combine the boundary list from the "min" approach (since that's simpler),
    # and see if there's a pair j, j+K with b[j+K] - b[j] == N. If yes, feasible.
    # We'll do binary search for x in [0.. S//K].
    #
    # Then to find the set of boundaries that can appear in some solution,
    # we use both min and max boundary lists. We look for sub-lists of length K
    # that satisfy b[j+K] - b[j] == N, gather those boundaries, and repeat
    # for all j. Finally, convert those boundary indices mod N to actual
    # cut lines in [0..N-1]. Output y = number of lines never used.
    #
    # ------------------------------------------------------------------

    # Fast prefix sum
    S = sum(A)
    max_possible = S // K  # upper bound on min partition sum
    # (The min possible is at least 0 or 1, but we know each piece >= 1, so
    #  the min partition sum >= 1 if K <= N. We can start from lo=1.)
    # If K=N, the best possible is min(A), but we'll just do the normal binary search anyway.

    # Build the doubled array
    A2 = A + A

    # Precompute prefix sums for A2 for quick range sums if needed
    # But we'll do streaming sums in a greedy approach, so might not store.

    def build_boundaries_min(x):
        """
        Build boundary array with "cut as soon as sum >= x".
        Return list of indices b, with b[0] = 0.
        Indices are from 0..2N (the position *after* reading 0 items to after reading 2N items).
        """
        b = [0]
        ssum = 0
        for i in range(2*N):
            ssum += A2[i]
            if ssum >= x:
                b.append(i+1)  # boundary after piece i
                ssum = 0
        return b

    def build_boundaries_max(x):
        """
        Build boundary array with "cut as late as possible".
        That is, we keep adding pieces while the current sum is still < x,
        we cannot cut yet. As soon as adding the next piece still keeps sum < x,
        we do so. Actually, the logic is: we must ensure each segment has sum >= x.
        The 'max' version is typically done from right to left or we do a small trick:
          - we can invert A2 into B2 = (x - A2[i]) and do min approach sum >= 0?
        It's easier to mirror the logic from the right side, but let's do a direct approach:
          We'll parse from left to right, but we move a pointer for the start of the segment
          only after we confirm the next piece would cause sum < x if we don't cut, etc.
        Actually, let's do a direct "reverse minimal" approach: we will go from 2N-1 down to 0
          and cut as soon as sum >= x, then reverse the boundary array.
        """
        b_rev = [2*N]
        ssum = 0
        for i in range(2*N-1, -1, -1):
            ssum += A2[i]
            if ssum >= x:
                b_rev.append(i)
                ssum = 0
        b_rev.reverse()
        return b_rev

    def can_partition_x(x):
        """
        Return True if we can partition the circular array A into K arcs
        each sum >= x.
        We'll use the 'min' boundary list approach and check if there's a
        subarray of length N that fits exactly K arcs. We look for consecutive
        K+1 boundaries b[j], b[j+K] with b[j+K] - b[j] == N.
        """
        # Quick check: if x > max(A), then we might fail trivially unless we combine pieces.
        # But let's just do the full check.

        b = build_boundaries_min(x)
        # We want to see if there is j s.t. j+K <= len(b)-1 and b[j+K] - b[j] == N
        # Because that means from boundary b[j] to b[j+K] (exclusive) is N pieces
        # forming exactly K arcs. (Between b[j]..b[j+1]-1 is the 1st arc, etc.)
        # We only need at least one such j.

        # We'll do a two-pointer approach in the boundary array.
        # Because b is strictly increasing, we can do:
        left = 0
        for right in range(len(b)):
            while b[right] - b[left] > N:
                left += 1
            # Now check if exactly N:
            if b[right] - b[left] == N:
                # The number of segments in [left..right-1] is (right - left)
                # Actually that forms (right - left) arcs. We want K arcs => right-left == K
                if (right - left) == K:
                    return True
        return False

    # Binary search for the largest x
    lo, hi = 1, max_possible
    answer_x = 1  # at least 1 is feasible if there's a single piece of mass >= 1
    while lo <= hi:
        mid = (lo + hi)//2
        if can_partition_x(mid):
            answer_x = mid
            lo = mid+1
        else:
            hi = mid-1

    # Now, answer_x is the maximum min sum. Next we find how many cut lines are NEVER used
    # under any optimal partition. We'll collect all cut lines that can appear in some
    # valid partition achieving answer_x. Then the rest are "never used."

    # We'll gather boundaries from both minimal and maximal boundary lists (on A2)
    # and look for sub-windows of length N that yield K arcs. Any boundary inside
    # such a sub-window belongs to some valid partition.

    b_min = build_boundaries_min(answer_x)
    b_max = build_boundaries_max(answer_x)
    # We'll unify them to reduce code duplication. We'll just process each of them
    # with the same procedure to gather "possible boundaries".

    possible_boundary_indices = set()

    def collect_boundaries(b_list):
        """
        For each pair j, j+K with b_list[j+K] - b_list[j] == N,
        we gather all boundary indices b_list[t] for t in [j..j+K].
        Because that sub-window from b_list[j]..b_list[j+K]-1 is a valid set
        of K arcs for the circular partition (once mapped mod N).
        """
        left = 0
        for right in range(len(b_list)):
            # Move left if difference is > N
            while b_list[right] - b_list[left] > N:
                left += 1
            # Now check if difference == N
            if b_list[right] - b_list[left] == N:
                # We have K arcs if (right-left) == K
                if (right - left) == K:
                    # gather boundaries from b_list[left..right]
                    for t in range(left, right+1):
                        possible_boundary_indices.add(b_list[t])
                    # we continue scanning
        # done

    collect_boundaries(b_min)
    collect_boundaries(b_max)

    # Now possible_boundary_indices is a set of indices in [0..2N].
    # Each boundary index i means "we cut right after piece (i-1)" (mod 2N).
    # But we only care about cuts mod N.  If i in [0..N], that corresponds
    # to cutting after piece (i-1 mod N). Similarly if i in [N+1..2N], we reduce i mod N.
    # We want to figure out which lines between piece j and j+1 are used:
    #   The line i is "cut line i" between piece i and i+1 (mod N, 1-based).
    #   In 0-based, it's the boundary between A[i] and A[i+1].
    #
    # If boundary index b means "the cut is just before item b in the 0-based array."
    # We interpret that modulo N to see which cut line that is in the circle of length N.
    #
    # For a boundary index b in [0..2N], the actual "cut line" in the circle is b mod N.
    # Because if b mod N = x, that means the cut is between piece (x-1) mod N and piece x.
    #
    # Special note: b=0 means a boundary before the first piece, which is the same as
    # the boundary between piece N-1 and piece 0 in 0-based (cut line (N-1) in 0-based).
    # So effectively, the cut line is (b mod N - 1) mod N in 0-based.  But more simply,
    # if we say "the cut line index" in [0..N-1], we can define it as (b-1) mod N.
    # The problem statement 1-based numbering: cut line i is between piece i and i+1 (1-based).
    #
    # Let's map boundary b -> cut line c = (b-1) mod N (0-based).
    # Then 0-based cut line c corresponds to problem's i if i = c+1 (1-based).
    #
    # We'll proceed in 0-based, then convert at the end if needed. We only need the count.
    # So let's gather all 0-based cut lines that appear:

    used_cut_lines = set()
    for b_idx in possible_boundary_indices:
        c = (b_idx - 1) % N  # 0-based line
        used_cut_lines.add(c)

    # The number of lines used in some solution is len(used_cut_lines).
    # The number never used is N - len(used_cut_lines).

    y = N - len(used_cut_lines)

    print(answer_x, y)


def main():
    solve()

if __name__ == "__main__":
    main()