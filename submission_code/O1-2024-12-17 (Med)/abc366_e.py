def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    D = int(input_data[1])
    xvals = []
    yvals = []
    pos = 2
    for _ in range(N):
        x = int(input_data[pos]); y = int(input_data[pos+1])
        pos += 2
        xvals.append(x)
        yvals.append(y)

    # Sort x and y separately
    xvals.sort()
    yvals.sort()

    # ----------------------------------------------------------------
    # STEP 1: Build histogram H of all possible "Sx(x)" values (where Sx(x) = sum of |x - x_i|)
    #         only for x in a certain bounding range, and only record counts where Sx(x) <= D.
    #
    # We will sweep x from xLow = (min_x - D) to xHigh = (max_x + D).
    # We'll maintain Sx(x) in O(1) increments as x increases by 1:
    #    Sx(x+1) = Sx(x) + (# of x_i <= x) - (# of x_i > x) = Sx(x) + 2*count_left - N
    #
    # Then we increment a histogram H[Sx(x)] for 0 <= Sx(x) <= D.
    # Finally, we convert H into a prefix-sum array so that H[c] = number of x with Sx(x) <= c.
    # ----------------------------------------------------------------

    minx = xvals[0]
    maxx = xvals[-1]
    xLow = minx - D
    xHigh = maxx + D
    range_x = xHigh - xLow + 1  # how many integer x we will sweep

    # Compute initial Sx(xLow)
    Sx_curr = 0
    for xx in xvals:
        Sx_curr += abs(xLow - xx)

    # count_left = number of x_i <= xLow
    idx = 0
    while idx < N and xvals[idx] <= xLow:
        idx += 1

    # H will hold counts of Sx(x) for x in the sweep
    # We only care about Sx(x) up to D, so H has size D+1
    H = [0]*(D+1)

    x_now = xLow
    for i in range(range_x):
        # record in histogram if within [0..D]
        if 0 <= Sx_curr <= D:
            H[Sx_curr] += 1

        if i == range_x - 1:
            # no next x to move to
            break

        # move from x to x+1
        # update Sx(x+1) = Sx_curr + (# of xvals <= x) - (# of xvals > x)
        # where (# of xvals <= x) = idx
        Sx_next = Sx_curr + (2*idx - N)
        Sx_curr = Sx_next

        x_now += 1
        # now increase idx while xvals[idx] == x_now (meaning xvals[idx] <= x_now)
        while idx < N and xvals[idx] <= x_now:
            idx += 1

    # Convert H into prefix sums so that H[c] = number of x for which Sx(x) <= c
    for i in range(1, D+1):
        H[i] += H[i-1]

    # ----------------------------------------------------------------
    # STEP 2: For each integer y in the bounding range [yLow..yHigh],
    #         compute Sy(y) = sum of |y - y_i|.  Then the condition
    #         Sx(x) + Sy(y) <= D  =>  Sx(x) <= D - Sy(y).
    #         The number of x with Sx(x) <= (D - Sy(y)) is H[D - Sy(y)] if (D - Sy(y)) >= 0.
    #
    # We similarly sweep y from yLow = (min_y - D) to yHigh = (max_y + D),
    # updating Sy(y) incrementally and adding to the answer.
    # ----------------------------------------------------------------

    miny = yvals[0]
    maxy = yvals[-1]
    yLow = miny - D
    yHigh = maxy + D
    range_y = yHigh - yLow + 1

    # initial Sy(yLow)
    Sy_curr = 0
    for yy in yvals:
        Sy_curr += abs(yLow - yy)

    # idy = number of yvals <= yLow
    idy = 0
    while idy < N and yvals[idy] <= yLow:
        idy += 1

    ans = 0
    y_now = yLow
    for i in range(range_y):
        # we can contribute if Sy_curr <= D
        if 0 <= Sy_curr <= D:
            ans += H[D - Sy_curr]

        if i == range_y - 1:
            break

        # update Sy(y+1)
        Sy_next = Sy_curr + (2*idy - N)
        Sy_curr = Sy_next

        y_now += 1
        # move idy while yvals[idy] <= y_now
        while idy < N and yvals[idy] <= y_now:
            idy += 1

    print(ans)