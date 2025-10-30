def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    s = next(it).strip()
    # cost[i] is the cost for flipping s[i]. (0-indexed)
    cost = [int(next(it)) for _ in range(n)]

    # Our strategy is to produce a string T from s by flipping some bits at a total cost.
    # We need T to be “good” – that is, among all adjacent pairs (T[i], T[i+1])
    # exactly one pair is equal, and all others are different.
    # One way to satisfy this is to “force” a single violation (i.e. an adjacent pair that are equal)
    # at a candidate boundary between indices i and i+1 (0-indexed) for some i from 0 to n-2,
    # while keeping all other adjacent pairs alternating.
    #
    # How to get T? Think of T as two pieces:
    # 1. Left part: indices 0..i
    #    We force T[0..i] to be strictly alternating.
    #    But we can choose the starting bit b (either '0' or '1').
    # 2. Right part: indices i+1..n-1
    #    We force T[i+1..n-1] to be strictly alternating with the twist that we require T[i+1]
    #    to equal T[i] (thus producing the one violation).
    #
    # Once b (the starting bit for the left segment) and violation index i is fixed,
    # the entire T is determined:
    # - For indices 0..i, T[j] = b if (j - 0) is even, else the complement.
    # - Then T[i] is determined by that pattern.
    # - And for indices i+1..n-1, we set T[i+1] = T[i] (forcing the violation),
    #   and then alternate: that is, for j >= i+1, T[j] = T[i] if (j-(i+1)) is even, otherwise its complement.
    #
    # Our goal is to choose i (violation position between i and i+1) and b in {'0','1'}
    # so that total cost (the sum of individual flipping costs when S and T differ) is minimized.
    #
    # The naive approach would try every candidate and recalc the cost in O(n) each,
    # resulting in O(n^2) time. With n up to 2e5, we need an O(n) or O(n log n) solution.
    #
    # We precompute the cost to change S over any segment into a prescribed alternating pattern.
    # For a segment [L, R] (inclusive) and a chosen starting bit x,
    # the required T is: for any j in [L, R]:
    #   desired = x if (j - L) is even, else the complement.
    #
    # Define a helper function seg_cost(L, R, x) that computes the cost to change S[L..R]
    # into such a pattern.
    #
    # To answer many seg_cost queries quickly, we precompute, for each index j, the cost to flip S[j]
    # to '0' and to '1'. Then we build cumulative sums separately for even and odd indices.
    
    N = n
    # Precompute flip costs per index to become '0' and '1'
    F0 = [0] * N  # cost to flip S[j] into '0'
    F1 = [0] * N  # cost to flip S[j] into '1'
    for j in range(N):
        F0[j] = cost[j] if s[j] != '0' else 0
        F1[j] = cost[j] if s[j] != '1' else 0

    # Build prefix sums for indices: we maintain separate cumulative sums for even and odd indices.
    # even0: cumulative sum of F0 at even indices.
    # odd0: cumulative sum of F0 at odd indices.
    # Similarly even1 and odd1.
    even0 = [0] * (N + 1)
    odd0 = [0] * (N + 1)
    even1 = [0] * (N + 1)
    odd1 = [0] * (N + 1)
    for j in range(N):
        even0[j + 1] = even0[j]
        odd0[j + 1] = odd0[j]
        even1[j + 1] = even1[j]
        odd1[j + 1] = odd1[j]
        if j % 2 == 0:
            even0[j + 1] += F0[j]
            even1[j + 1] += F1[j]
        else:
            odd0[j + 1] += F0[j]
            odd1[j + 1] += F1[j]

    # seg_cost(L, R, x) computes the cost to transform S[L..R] into an alternating pattern that starts with bit x.
    # For any j in [L, R]:
    #   if (j - L) is even, desired = x.
    #   if (j - L) is odd, desired = the opposite of x.
    def seg_cost(L, R, x):
        # We use our cumulative sums. Note that "parity" in the segment is defined relative to L.
        # When L is even, indices j that are even relative to 0 are even offsets.
        # When L is odd, indices j that are odd relative to 0 are even offsets.
        if x == '0':
            # Even offset positions: want '0'
            # Odd offset positions: want '1'
            if L % 2 == 0:
                # even indices in [L,R] are positions with even offset.
                cost_even = even0[R + 1] - even0[L]
                cost_odd = odd1[R + 1] - odd1[L]
            else:
                # If L is odd, then indices with odd j are even offsets, and even j are odd offsets.
                cost_even = odd0[R + 1] - odd0[L]
                cost_odd = even1[R + 1] - even1[L]
            return cost_even + cost_odd
        else:
            # x == '1'
            # Even offset positions: want '1'
            # Odd offset positions: want '0'
            if L % 2 == 0:
                cost_even = even1[R + 1] - even1[L]
                cost_odd = odd0[R + 1] - odd0[L]
            else:
                cost_even = odd1[R + 1] - odd1[L]
                cost_odd = even0[R + 1] - even0[L]
            return cost_even + cost_odd

    # Now, consider each possible violation boundary i (0-indexed, so between index i and i+1, i=0..N-2)
    # and each possible starting bit b for the left segment.
    # For the left segment (indices 0..i), we want an alternating pattern starting with b.
    # The cost for the left segment is seg_cost(0, i, b).
    # Let T[i] (the last element of the left segment) be L_val. It is determined by:
    #    L_val = b if (i - 0) is even, else the opposite, i.e.,
    #           L_val = b if i is even, else ( '1' if b == '0' else '0' )
    # Then, for the right segment (indices i+1..N-1), we want an alternating pattern
    # that starts with L_val – so that T[i+1] = L_val (thus T[i]==T[i+1]) producing our one violation.
    #
    # The total cost is left_cost + right_cost.
    #
    # We search for the candidate (i, b) that minimizes that total cost.
    INF = 10**20
    best = INF
    for i in range(0, N - 1):
        for b in ['0', '1']:
            left_cost = seg_cost(0, i, b)
            # Compute T[i] from left segment.
            if i % 2 == 0:
                L_val = b
            else:
                L_val = '1' if b == '0' else '0'
            right_cost = seg_cost(i + 1, N - 1, L_val)
            total_cost = left_cost + right_cost
            if total_cost < best:
                best = total_cost

    sys.stdout.write(str(best))


if __name__ == "__main__":
    main()