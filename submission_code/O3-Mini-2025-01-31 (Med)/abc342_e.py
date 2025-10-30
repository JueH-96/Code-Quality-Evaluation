def main():
    import sys,heapq
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    
    # Build reverse graph:
    # For each train‐info (l, d, k, c, A, B), the trains run on times:
    #   t = l, l+d, l+2d, …, l+(k-1)*d
    # and the train goes from A to B (depart at t, arrive at t+c).
    # In the backward dynamic program we want to “relax” from station B to station A.
    # So for each piece of information we add an edge from B to A (with parameters) to represent that
    # if you can be at B at time T_B then you can be at A at a time given by the best train in this group.
    rev = [[] for _ in range(n+1)]
    for _ in range(m):
        l = int(next(it))
        d = int(next(it))
        k = int(next(it))
        c = int(next(it))
        A = int(next(it))
        B = int(next(it))
        # Reverse the edge: from station B to station A.
        rev[B].append((A, l, d, k, c))
    
    # Let avail[s] be the maximum time at which one can be at station s so that there exists
    # a valid train sequence from s to station n.
    # For station n, we set avail[n] = INF as an initial condition.
    # (Though technically if you’re already at station n you have “arrived”.)
    INF = 10**20  # A value larger than any possible departure time (~ 1e18)
    avail = [-1]*(n+1)
    avail[n] = INF

    # We will use a priority queue (max-heap) to process stations in order of decreasing avail time.
    # In Python, heapq is a min-heap so we push negative values.
    heap = []
    heapq.heappush(heap, (-avail[n], n))
    
    # Process the reverse edges.
    while heap:
        cur_val_neg, station = heapq.heappop(heap)
        cur_val = -cur_val_neg
        # We skip if this isn't the latest value for this station.
        if cur_val != avail[station]:
            continue
        # For each train that arrives at 'station' (i.e. edge from A -> station)
        for (A, l, d, k, c) in rev[station]:
            # This train info gives that a train departs from A at some time t in:
            #   t = l, l+d, l+2d, …, l+(k-1)*d,
            # and arrives at station at time t+c.
            # A connection is possible if t+c <= avail[station].
            # Hence we need t <= avail[station] - c.
            # Let limit = avail[station] - c.
            limit = cur_val - c
            if limit < l:
                # Even the earliest train departs too late relative to limit.
                continue
            # We want the maximum allowed departure time from A in this group.
            # That is the largest t of the form t = l + x*d (with 0 <= x < k) that is <= limit.
            # Compute candidate = l + d * floor((limit-l)/d) and cap it by l+(k-1)*d.
            cand = l + d * ((limit - l) // d)
            max_possible = l + (k - 1) * d
            if cand > max_possible:
                cand = max_possible
            # If we can improve avail[A] with this candidate departure time, update it.
            if cand > avail[A]:
                avail[A] = cand
                heapq.heappush(heap, (-cand, A))
    
    # Print the answers for stations 1 to n-1.
    # If avail[s] is still -1 then no sequence is possible from s.
    out_lines = []
    for i in range(1, n):
        if avail[i] < 0:
            out_lines.append("Unreachable")
        else:
            out_lines.append(str(avail[i]))
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()