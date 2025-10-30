def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    idx = 2

    # edges_in[ b ] = list of (a, l, d, k, c) meaning
    # there is a train from a -> b that starts at times l, l+d, ..., l+(k-1)*d
    # and takes c time to go from a to b
    # We'll store these so we can work "backwards" from b to a.
    edges_in = [[] for _ in range(N+1)]
    for _ in range(M):
        l = int(input_data[idx]);     idx += 1
        d = int(input_data[idx]);     idx += 1
        k = int(input_data[idx]);     idx += 1
        c = int(input_data[idx]);     idx += 1
        A = int(input_data[idx]);     idx += 1
        B = int(input_data[idx]);     idx += 1
        edges_in[B].append((A, l, d, k, c))

    # We'll compute f[i] = the maximum time one can depart station i and still reach station N
    # If unreachable, f[i] = None
    # We use a large constant to represent "infinity".
    INF = 10**20
    f = [None]*(N+1)
    f[N] = INF  # We can "arrive" at station N at arbitrarily large time, so set it to "infinite".

    # We'll use a max-heap by pushing negative values (Python's heapq is a min-heap).
    pq = []
    heapq.heappush(pq, (-f[N], N))  # (priority, station)

    while pq:
        # current largest f-value
        cur_val, station = heapq.heappop(pq)
        cur_val = -cur_val

        # If this is stale (we have a newer, larger f-value for 'station'), skip
        if f[station] != cur_val:
            continue

        # Relax edges leading into 'station'
        for (prev_station, l, d, k, c) in edges_in[station]:
            # We want T + c <= f[station]
            # T must be one of l, l+d, l+2d, ..., l+(k-1)*d
            # So T <= f[station] - c
            # We'll compute the largest T that satisfies those constraints.
            if cur_val == INF:
                # If we can arrive at 'station' arbitrarily late,
                # the best T for 'prev_station' is simply the last possible departure time
                # which is l + (k - 1)*d, provided k>0. (Constraints ensure k>=1)
                T = l + (k-1)*d
            else:
                # We require: T <= f[station] - c
                limit = cur_val - c - l
                if limit < 0:
                    # No valid departure time
                    continue
                # T' = floor( limit / d )
                Tprime = limit // d
                if Tprime < 0:
                    continue
                if Tprime >= k:
                    Tprime = k-1
                T = l + Tprime*d

            # T can't be negative under problem constraints (l>0), but let's be safe:
            if T < l:
                # If T came out below the first valid departure, it's invalid
                continue

            # Now attempt to update f[prev_station]
            if f[prev_station] is None or T > f[prev_station]:
                f[prev_station] = T
                heapq.heappush(pq, (-T, prev_station))

    # Print results for f(1), f(2), ..., f(N-1)
    # If f[i] is None, print "Unreachable" else print f[i].
    # Note that the problem statement wants exactly N-1 lines.
    out = []
    for i in range(1, N):
        if f[i] is None:
            out.append("Unreachable")
        else:
            out.append(str(f[i]))
    print("
".join(out))

# Do not forget to call main()
if __name__ == "__main__":
    main()