def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Faster output
    import sys
    out = sys.stdout.write

    # Parse inputs
    N = int(input_data[0])
    M = int(input_data[1])
    X1_forced = int(input_data[2])

    # Trains: each entry will be (A_i,B_i,S_i,T_i,original_index)
    trains = []
    pt = 3
    for i in range(1, M+1):
        A_i = int(input_data[pt]);   B_i = int(input_data[pt+1])
        S_i = int(input_data[pt+2]); T_i = int(input_data[pt+3])
        trains.append((A_i, B_i, S_i, T_i, i))
        pt += 4

    # We will store the delay X_i in an array X of size M+1 (1-based index)
    # X[train_index] = delay for that train
    X = [0]*(M+1)
    # Train 1 has X_1 = X1_forced
    # (We don't yet know which of the input lines is "train 1" until we see i=1.)
    # We will assign that after reading them.

    # Group by city for building Fenwicks:
    # city_map[c] = {
    #    "arrivals": [(T_i, index_in_trains), ...],     # for trains that end in city c
    #    "departures": [(S_i, index_in_trains), ...],   # for trains that start in city c
    #    "times": sorted unique times (all T_i from arrivals and S_i from departures),
    #    "mapTime": { time_value -> compressed_index },
    #    "fenwicks": Fenwicks structure
    # }
    from collections import defaultdict
    city_map = defaultdict(lambda: {"arrivals":[], "departures":[]})

    # Fill city_map
    for (a,b,s,t,i) in trains:
        city_map[a]["departures"].append((s, i))
        city_map[b]["arrivals"].append((t, i))

    # Build coordinate-compression and Fenwicks for each city
    # We only ever query or update with times up to "S_i" or exactly "T_i".
    # For city c, collect all times from arrivals (T_i) + departures (S_i).
    for c in city_map.keys():
        arr_times = [ x[0] for x in city_map[c]["arrivals"] ]
        dep_times = [ x[0] for x in city_map[c]["departures"] ]
        all_times = arr_times + dep_times
        all_times = list(set(all_times))
        all_times.sort()
        city_map[c]["times"] = all_times
        # Build mapTime
        mp = {}
        for idx, val in enumerate(all_times):
            mp[val] = idx
        city_map[c]["mapTime"] = mp
        # Fenwicks array for maximum queries, size = len(all_times)
        size = len(all_times)
        # We'll store Fenwicks as a list "fenwicks" where fenwicks[i] = current max for prefix i
        # We'll implement them in 1-based indexing internally.
        fenwicks = [0]* (size+1)
        city_map[c]["fenwicks"] = fenwicks

    # Fenwicks helper functions
    def fenwicks_update(fw, i, val):
        # fw is 1-based inside, i is 0-based from compression
        i += 1
        while i < len(fw):
            if fw[i] < val:
                fw[i] = val
            i += i & -i

    def fenwicks_query(fw, i):
        # max over [0..i], i is 0-based
        # returns maximum in fenwicks up to i
        i += 1
        s = 0
        while i > 0:
            if fw[i] > s:
                s = fw[i]
            i -= i & -i
        return s

    # We will sort all trains by S_i ascending and process in that order.
    # This way, any train j with S_j < S_i (processed earlier) will have
    # already updated the Fenwicks for the city B_j with X_j+T_j,
    # enabling queries for trains i that start later.
    trains_sorted_by_S = sorted(trains, key=lambda x: x[2])  # (A,B,S,T,i) sorted by S

    # We'll store A,B,S,T in arrays for easier indexing
    # so we can quickly find them by original index i.
    A = [0]*(M+1)
    B = [0]*(M+1)
    S = [0]*(M+1)
    T = [0]*(M+1)
    for (a,b,s,t,i) in trains:
        A[i] = a
        B[i] = b
        S[i] = s
        T[i] = t

    # Identify which input line is "train 1" (the one with i=1) and set X[1]=X1_forced
    # Actually from the problem statement, "Train i" means the i-th line is train i.
    # So the line with i=1 is just trains[...] where the last field = 1. That is indeed train 1.
    # We'll just do: X[1] = X1_forced at the end of reading, that is correct.
    X[1] = X1_forced

    # Pre-insert train 1's Fenwicks update when we reach it in the loop, not before.
    # Because we unify the process for all trains in ascending S_i.

    # Process in ascending S. For each train i:
    #   c = A[i]
    #   # query fenwicks in city c up to time S[i]
    #   val = fenwicks_query(..., posS)
    #   if i != 1: X[i] = max(0, val - S[i])  (since X[1] is forced)
    #   # then update fenwicks in city B[i] at T[i] with X[i] + T[i]
    #   fenwicks_update(..., posT, X[i] + T[i])

    for (a,b,s,t,i) in trains_sorted_by_S:
        # Query in city a
        c = a
        pos_s = city_map[c]["mapTime"][s]
        fw = city_map[c]["fenwicks"]
        val = fenwicks_query(fw, pos_s)

        if i != 1:
            # Set X[i] to the minimum feasible that satisfies constraints
            cand = val - s
            if cand < 0:
                cand = 0
            X[i] = cand
        else:
            # i == 1 => forced delay
            pass

        # Now update city b Fenwicks with X[i] + T[i]
        c2 = b
        pos_t = city_map[c2]["mapTime"][t]
        fw2 = city_map[c2]["fenwicks"]
        up_val = X[i] + t
        # update
        fenwicks_update(fw2, pos_t, up_val)

    # Finally, output X[2..M]
    # Print them separated by spaces
    out(" ".join(str(X[i]) for i in range(2, M+1)) + "
")