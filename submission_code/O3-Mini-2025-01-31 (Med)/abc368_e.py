def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))
    given_X1 = int(next(it))
    
    # We represent each train as a tuple: (train_id, A, B, S, T)
    trains = [None] * M
    for i in range(M):
        A = int(next(it))
        B = int(next(it))
        S = int(next(it))
        T = int(next(it))
        trains[i] = (i + 1, A, B, S, T)
    
    # X[i] will store the delay for train i (1-indexed).
    # Train 1's delay is fixed to the given value.
    X = [0] * (M + 1)
    
    # The condition for transfers (if train i can transfer to j) is:
    #   T_i + X_i <= S_j + X_j   when B_i == A_j and T_i <= S_j.
    # Rearranging, we have: X_j >= X_i + (T_i - S_j).
    # For a given city k, note that any arriving train i (with arrival time T_i and computed delay X_i)
    # offers a 'potential' value (T_i + X_i). And for a departing train j from city k (departure time S_j),
    # the transfer constraint implies:
    #    X_j >= (max_{i arriving to k with T_i <= S_j} (T_i + X_i)) - S_j.
    #
    # Because we want the overall sum of extra delays (except X1) to be minimum,
    # we assign each train j the minimum possible delay satisfying:
    #    X_j = max(0, max_{i arriving to city A_j with arrival time <= S_j} (T_i + X_i) - S_j)
    #
    # Since any valid transfer must have S_i < T_i and T_i <= S_j, the dependency from train i to j is acyclic.
    # Thus we process trains in order of their departure times S_j.
    
    # Build two orderings:
    # 1. Order trains by departure time S.
    dep_sorted = sorted(trains, key=lambda rec: rec[3])  # rec = (id, A, B, S, T)
    # 2. Order trains by arrival time T.
    arr_sorted = sorted(trains, key=lambda rec: rec[4])
    
    # For each city, we maintain a "potential" value:
    #   potential[city] = max( T_i + X_i )
    # among trains that have already arrived (i.e. with T_i less than or equal to the current S).
    NEG_INF = -10**18
    city_potential = [NEG_INF] * (N + 1)
    
    # Pointer for arrivals: as we increase S (departure times) we “release” trains that have arrived.
    arr_ptr = 0
    arr_count = len(arr_sorted)
    
    # Process each train in increasing order of departure time.
    for rec in dep_sorted:
        idx, A, B, S, T = rec
        
        # Update city potentials for trains that have arrived (T <= current departure time S).
        while arr_ptr < arr_count and arr_sorted[arr_ptr][4] <= S:
            tid, Ai, Bi, Si, Ti = arr_sorted[arr_ptr]
            # At this point, train tid has already been processed in our departure order
            # (since S_tid < T_tid <= S) so X[tid] is known.
            val = Ti + X[tid]
            if val > city_potential[Bi]:
                city_potential[Bi] = val
            arr_ptr += 1
        
        # For train 1, we simply use the given delay.
        if idx == 1:
            X[idx] = given_X1
        else:
            # For a train departing from city A at time S,
            # its delay must be at least: max(0, (max arrival potential at A) - S).
            required = city_potential[A] - S
            if required < 0:
                required = 0
            X[idx] = required
        
    # Output the answer: delays for trains 2,3,...,M in order of their train indices.
    output = []
    for i in range(2, M + 1):
        output.append(str(X[i]))
    sys.stdout.write(" ".join(output))

if __name__ == '__main__':
    main()