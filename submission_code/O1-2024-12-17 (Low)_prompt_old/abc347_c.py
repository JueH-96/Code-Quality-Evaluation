def solve():
    import sys

    data = sys.stdin.read().strip().split()
    N, A, B = map(int, data[:3])
    D = list(map(int, data[3:]))

    # The length of one "week"
    T = A + B

    # We will use the fact that if we can rotate the week so that all
    # D_i end up in the holiday segment [0..A-1] (in 0-based), then the
    # set of values (D_i - D_1) mod T must lie entirely in some segment
    # of length A on the circular track of length T.
    #
    # A known criterion: If the maximum gap between consecutive points
    # in the sorted set (including wrap-around) is G, then all points
    # fit into a (T-G)-length arc. Therefore, we can fit all points
    # into an arc of length A if and only if (T - G) < A.
    #
    # Steps:
    #  1) Fix D_1 as a reference. Let d_i = (D_i - D_1) mod T.
    #  2) Sort d_i.
    #  3) Find the largest gap G between consecutive d_i (including wrap).
    #  4) Check if T - G < A => "Yes", else "No".

    # Reference day
    base = D[0]

    # Compute d_i = (D_i - base) mod T
    d = [(x - base) % T for x in D]

    d.sort()

    # Compute the largest gap between consecutive sorted d
    largest_gap = 0
    for i in range(len(d) - 1):
        gap = d[i+1] - d[i]
        if gap > largest_gap:
            largest_gap = gap

    # Also include the wrap-around gap from last back to first
    wrap_gap = T - d[-1] + d[0]  # but d[0] should be 0 in principle
    if wrap_gap > largest_gap:
        largest_gap = wrap_gap

    # If the points can fit in an arc of length A, we need (T - largest_gap) < A
    if T - largest_gap < A:
        print("Yes")
    else:
        print("No")


# Call solve() to execute
solve()