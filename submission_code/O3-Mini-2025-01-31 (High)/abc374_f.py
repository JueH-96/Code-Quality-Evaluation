def main():
    import sys
    # Read input values.
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    X = int(next(it))
    # The order placement times; they are non‐decreasing.
    T = [int(next(it)) for _ in range(n)]

    # We will use dynamic programming.
    # Let dp[i] be the minimum total dissatisfaction (cost) to ship orders 0..i-1 (i orders)
    # and let last_day[i] be the shipping day at which the last shipment (that covered orders 0..i-1)
    # was made. (This shipping day is chosen as low as possible, because shipping later only increases cost.)
    #
    # When we ship a group that covers orders j .. i-1 (for some j with 0 ≤ j < i), note that
    # the orders are sorted so that the maximum T in that group is T[i-1].
    # Also, the rule "at most K orders can be shipped together" forces (i - j) ≤ k.
    #
    # The allowed shipping day for the new group is:
    #   S = max(last_day[j] + X, T[i-1])
    # (For j == 0 we let last_day[0] be a very small number so that S = T[i-1].)
    #
    # Then the cost incurred for shipping orders j .. i-1 together at day S is:
    #   cost_group = (i-j)*S - (sum of T[j] through T[i-1])
    #
    # Our recurrence is:
    #   dp[i] = min_{j from max(0, i-k) to i-1} { dp[j] + (i-j)*S - (prefix[i] - prefix[j]) }
    # where S = max(last_day[j] + X, T[i-1]).
    
    # Set up an array for dp and last_day.
    INF = 10**30  # a large number (larger than any possible cost)
    dp = [0] * (n + 1)
    last_day = [0] * (n + 1)
    dp[0] = 0
    # for the base state, we set last_day[0] to a very small value.
    last_day[0] = -10**18

    # Precompute prefix sums: p[i] = T[0] + T[1] + ... + T[i-1]
    p = [0] * (n + 1)
    for i in range(1, n + 1):
        p[i] = p[i - 1] + T[i - 1]

    # Process orders 1 through n (dp[i] is best cost to cover first i orders)
    for i in range(1, n + 1):
        best = None
        best_ship = None
        # The group we ship in the last shipment can contain at most k orders so j can range from i-k to i.
        j_start = max(0, i - k)
        for j in range(j_start, i):
            # For orders j ... i-1, the required shipping day S must be at least:
            #   (a) last_day[j] + X   (the waiting period after the previous shipment)
            #   (b) T[i-1]            (an order cannot be shipped before its placement and T[i-1] is maximum in group)
            # Hence choose S = max(last_day[j] + X, T[i-1])
            S = T[i - 1] if last_day[j] + X <= T[i - 1] else last_day[j] + X
            # Compute cost for shipping group j...i-1 at day S.
            # Group cost = (number of orders) * S - (sum of placement times in the group)
            cost_group = (i - j) * S - (p[i] - p[j])
            candidate = dp[j] + cost_group
            # In case of tie in cost, choose the grouping with the smaller shipping day S,
            # since that gives a lower “last shipment day” and can be beneficial later.
            if best is None or candidate < best:
                best = candidate
                best_ship = S
            elif candidate == best:
                if S < best_ship:
                    best_ship = S
        dp[i] = best
        last_day[i] = best_ship

    sys.stdout.write(str(dp[n]))

if __name__ == '__main__':
    main()