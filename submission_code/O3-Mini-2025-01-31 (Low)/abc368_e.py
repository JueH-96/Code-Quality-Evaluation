def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    it = iter(input_data)
    n = int(next(it))
    m = int(next(it))
    X1 = int(next(it))
    # We'll use 1-indexing for trains.
    trains = [None] * (m + 1)
    for i in range(1, m + 1):
        A = int(next(it))
        B = int(next(it))
        S = int(next(it))
        T = int(next(it))
        trains[i] = (A, B, S, T)

    # dp[i] represents the delay assigned to train i.
    # It must be at least 0. We are given X1 = dp[1].
    dp = [0] * (m + 1)
    dp[1] = X1

    # Our constraint is: for any two trains i -> j (transfer is possible originally)
    # we require: T_i + X_i <= S_j + X_j.
    # Rearranging, X_j >= X_i + (T_i - S_j).
    #
    # In a minimal delay assignment, we want to let dp[j] be as small as possible.
    # Note that since originally T_i <= S_j (transfer possibility), the quantity (T_i - S_j)
    # is non-positive, so many constraints are automatically satisfied if dp[j] is 0.
    # However, some delays may be forced by earlier delays.
    #
    # The key observation is that for a train j departing from city c at time S_j,
    # for any train i arriving at the same city c at time T_i (with T_i <= S_j) we must have:
    #      dp[j] >= dp[i] + (T_i - S_j).
    # Equivalently, dp[j] must be at least (dp[i] + T_i) - S_j.
    #
    # Thus, if for city c we maintain the best value of (dp[i] + T_i) among all trains
    # which have already arrived (i.e. have T_i less than the current departure time),
    # then at the departure event of train j (departing from city c at time S_j) the delay
    # must satisfy:
    #      dp[j] >= max(0, (max_{i with arrival at c before S_j} (dp[i] + T_i)) - S_j).
    #
    # We can process a mixed list of departure and arrival events in increasing order of time:
    # - When a train departs (at time S), we update its dp from the best candidate at its departure city.
    # - When a train arrives (at time T), we update a "best" value for its arrival city, which is:
    #       best[city] = max(previous value, dp[i] + T)
    #
    # We build an event list for all trains.
    # Each train i will have a departure event at time S_i and an arrival event at time T_i.
    # Note: For each train, S_i < T_i so we know the departure comes before arrival.
    #
    # We'll maintain an array "best" for each city, initially very small.
    infneg = -10**18
    best = [infneg] * (n + 1)

    # Build event list: each event is (time, type, index)
    # We'll use type 0 for departure events and type 1 for arrival events.
    events = []
    for i in range(1, m + 1):
        A, B, S, T = trains[i]
        events.append((S, 0, i))
        events.append((T, 1, i))
    events.sort()  # sorts by time, and since S < T for each train, their order is proper

    # Process events in increasing order.
    for time, typ, i in events:
        A, B, S, T = trains[i]
        if typ == 0:  # Departure event for train i at time S.
            if i == 1:
                # dp[1] is given.
                pass
            else:
                # The best candidate from arrivals at city A that occurred earlier:
                # candidate = best[A] - S.
                cand = best[A] - S
                if cand < 0:
                    cand = 0
                if cand > dp[i]:
                    dp[i] = cand
        else:  # Arrival event for train i at time T.
            # When train i arrives at city B at time T, update best[B]:
            val = dp[i] + T
            if val > best[B]:
                best[B] = val

    # We are asked to print delays X_2,...,X_M.
    # dp[1] corresponds to X_1 which is given.
    output = []
    for i in range(2, m + 1):
        output.append(str(dp[i]))
    sys.stdout.write(" ".join(output))


if __name__ == "__main__":
    main()