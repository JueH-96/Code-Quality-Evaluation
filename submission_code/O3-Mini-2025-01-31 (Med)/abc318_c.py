def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return

    # Parse input parameters
    it = iter(input_data)
    N = int(next(it))
    D = int(next(it))
    P = int(next(it))
    fares = [int(next(it)) for _ in range(N)]

    # Sort the fares in descending order so that we consider covering the most expensive days first.
    fares.sort(reverse=True)

    # Build a prefix sum array for the sorted fares. 
    # prefix[i] will hold the sum of the first i fares.
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i + 1] = prefix[i] + fares[i]

    # Total fare if we use no passes at all.
    total_fare = prefix[N]

    # We determine the maximum number of batches we might consider:
    # If we want to cover all days, we need at least ceil(N / D) batches.
    max_batches = (N + D - 1) // D

    # Try every possible number of batches from 0 up to max_batches.
    # Note that buying extra passes which cover more days than N is allowed,
    # but will not reduce the cost further once all days are covered.
    best = total_fare  # Starting with the cost without passes (m = 0)
    for m in range(0, max_batches + 1):
        # With m batches, we have m * D passes available.
        # However, we can cover at most N days.
        covers = m * D
        if covers > N:
            covers = N

        # Cost breakdown:
        # 1) Cost of buying m batches of passes: m * P
        # 2) For the days not covered by passes, we have to pay the regular fare.
        #    Since we cover the highest fares, the sum of regular fares for uncovered days is:
        #    total_fare - prefix[covers]
        cost = m * P + (total_fare - prefix[covers])
        best = min(best, cost)

    sys.stdout.write(str(best))

if __name__ == '__main__':
    main()