def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    T = int(input_data[1])
    S = input_data[2]
    X = list(map(int, input_data[3:]))

    # We want to count the number of pairs (i, j) that "pass" each other
    # between time 0 and time T+0.1.  Ant i passes ant j if and only if:
    #
    #   (1) They are initially in the order X_i < X_j, but i moves right (+1) and j moves left (-1),
    #       so that eventually i and j meet at time ((X_j - X_i) / 2) if 0 <= that time <= T+0.1.
    #
    #   OR, if we interpret i's position > j's, but i moves left and j moves right, they may
    #   cross as well.  However, it's simpler to catch them all by *sorting the ants by position*
    #   and then only looking for (right-moving, left-moving) across the sorted positions in ascending X.
    #
    # The time at which a right-moving ant at x_r and a left-moving ant at x_l (with x_r < x_l) cross is
    #       t_cross = (x_l - x_r) / 2
    # We require 0 <= t_cross <= T + 0.1.
    # That is 0 <= x_l - x_r <= 2(T + 0.1).
    #
    # So we only count pairs (r, l) with r < l in coordinate (i.e., x_r < x_l), direction_r = +1, direction_l = -1,
    # and (x_l - x_r) <= 2(T+0.1).
    #
    # We'll do:
    #  1) Sort ants by ascending X.
    #  2) Collect posR = positions of ants going right, in ascending order.
    #     Collect posL = positions of ants going left, in ascending order.
    #  3) Use a two-pointer approach to count how many l's are within distance 2(T+0.1) from each r,
    #     provided x_l > x_r.
    #
    # Because T can be up to 1e9 and X_i can be up to ±1e9, we will do an integer-based check
    # to avoid floating imprecision.  We'll multiply each position by 10, and compare against
    # crossingDistInt = 2*(T+0.1)*10 = 20*T + 2.  Then the condition becomes:
    #
    #       10*x_l - 10*x_r <= (20*T + 2)
    #
    # while still requiring x_l > x_r, i.e. 10*x_l > 10*x_r.  This is safe within 64-bit range
    # since 10*X_i can be up to 1e10 in absolute value, which is well within 2^63.

    # Prepare data
    ants = [(X[i], 1 if S[i] == '1' else -1) for i in range(N)]
    # Sort by position
    ants.sort(key=lambda x: x[0])

    # Build two arrays of positions (multiplied by 10) depending on direction
    posR = []  # right-moving
    posL = []  # left-moving
    for (pos, d) in ants:
        if d == 1:
            posR.append(pos * 10)
        else:
            posL.append(pos * 10)

    # We only need to count pairs (r, l) with r < l in coordinate sense.  In sorted order,
    # r < l means posR[i] < posL[j].  We'll keep two pointers:
    #   Lstart to skip all left-movers that are not strictly > r
    #   Lend   to find all left-movers that are within distance crossingDistInt of r.
    crossingDistInt = 20 * T + 2  # integer threshold for (x_l*10 - x_r*10)

    result = 0
    Lstart = 0  # index into posL
    Lend = -1   # index into posL

    lenL = len(posL)

    for r in posR:
        # move Lstart so that posL[Lstart] > r
        while Lstart < lenL and posL[Lstart] <= r:
            Lstart += 1

        # move Lend so that posL[Lend] - r <= crossingDistInt
        while Lend + 1 < lenL and (posL[Lend + 1] - r) <= crossingDistInt:
            Lend += 1

        if Lend >= Lstart:
            # all L's from Lstart..Lend (inclusive) satisfy the condition
            result += (Lend - Lstart + 1)

    print(result)

# Do not forget to call main()!
if __name__ == "__main__":
    main()