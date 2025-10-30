import sys

# Function to read input and solve the problem
def solve():
    # Read N, A, B from the first line
    # N: number of plans
    # A: number of holiday days (1-based: 1 to A)
    # B: number of weekday days (1-based: A+1 to A+B)
    N, A, B = map(int, sys.stdin.readline().split())

    # Read the list of plan days D from the second line
    # D_i: the i-th plan is scheduled D_i days later
    D = list(map(int, sys.stdin.readline().split()))

    # The length of a week
    # Week days can be indexed 0 to W-1.
    # Holidays are days 0 to A-1. Weekdays are A to W-1.
    W = A + B

    # Calculate the day of the week for each plan relative to a starting day 0.
    # If today is day 0, the i-th plan is on day D_i mod W.
    # We only care about the unique days of the week the plans could fall on
    # relative to some arbitrary starting day.
    # If today is day p (0-indexed), the i-th plan is on day (p + D_i) mod W.
    # The set of possible plan days { (p + D_i) mod W | i=1..N } depends on p.
    # The set of values { D_i mod W | i=1..N } represents the positions of the plans
    # relative to 'today' if 'today' was day 0.
    # Finding a p such that all (p + D_i) mod W are in [0, A-1] is equivalent
    # to finding a shift p such that the set {(p + d_i) mod W | d_i = D_i mod W}
    # is a subset of [0, A-1].
    # This is possible iff the set of points {D_i mod W} on the circle [0, W-1]
    # can be rotated (shifted by p) such that all rotated points fall within [0, A-1].
    # This is possible iff the minimum length of an interval on the circle that contains all
    # unique points {D_i mod W} is less than or equal to A-1.

    # Find the unique values of D_i mod W.
    d_mod_w = set()
    for d in D:
        d_mod_w.add(d % W)

    # Convert the set of unique values to a sorted list.
    # These are the unique positions on the week cycle relative to day 0.
    u = sorted(list(d_mod_w))
    M = len(u) # Number of unique D_i mod W values

    # If there's only one unique day modulo W (M=1), the points on the circle
    # consist of just that single point. The "gap" covering the rest of the circle is W.
    if M == 1:
        max_gap = W
    else:
        # Calculate the gaps between consecutive sorted unique values on the circle [0, W-1].
        # For M unique points on a circle, there are M gaps.
        # The first M-1 gaps are the differences between consecutive sorted points.
        gaps = []
        for i in range(M - 1):
            gaps.append(u[i+1] - u[i])

        # The last gap is the wrap-around distance from the last point (u[M-1])
        # back to the first point (u[0]), going through the W-1 -> 0 boundary.
        # On the number line, the distance from u[M-1] to u[0] in the next cycle is (u[0] + W) - u[M-1].
        wrap_around_gap = (u[0] + W) - u[M-1]
        gaps.append(wrap_around_gap)

        # The minimum length of an interval on the circle that contains all
        # unique points u_j is W - max_gap, where max_gap is the largest gap
        # between consecutive sorted unique points.
        max_gap = max(gaps)

    # The set of unique points {u_j} can be shifted into the interval [0, A-1]
    # (which has length A) if and only if the minimum length required to cover
    # these points on the circle (W - max_gap) is less than or equal to A-1.
    # Condition: W - max_gap <= A - 1
    # Rearranging the inequality: W - A + 1 <= max_gap
    # Since W = A + B, W - A = B.
    # The condition simplifies to B + 1 <= max_gap.
    # It is possible for all plans to be scheduled on holidays if and only if
    # the maximum gap between consecutive unique D_i mod W values on the circle
    # is greater than or equal to B + 1.

    if max_gap >= B + 1:
        print("Yes")
    else:
        print("No")

# Call the solve function to run the program
solve()