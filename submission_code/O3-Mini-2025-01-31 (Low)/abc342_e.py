def main():
    import sys, heapq
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    # We build the reverse graph.
    # For a train described by (l, d, k, c, A, B), a train runs from A to B.
    # When doing DP backward (from destination N), we want for each train arriving at v (B)
    # to update the possibility for its departure station u (A).
    # Thus, we add an entry for v containing (u, l, d, k, c).
    rev = [[] for _ in range(N+1)]
    for _ in range(M):
        l  = int(next(it))
        d  = int(next(it))
        k  = int(next(it))
        c  = int(next(it))
        A  = int(next(it))
        B  = int(next(it))
        rev[B].append((A, l, d, k, c))
      
    # Let dp[x] be the maximum departure time possible from station x that still leads to station N.
    # For station N, we can think of the "waiting time" as unbounded; we set dp[N] = INF.
    # For others, initialize dp[x] = -infty (meaning unreachable).
    INF = 10**30   # sufficiently large.
    NEG = -10**30
    dp = [NEG] * (N+1)
    dp[N] = INF
    
    # We use a priority queue (max-heap by dp value) for a Dijkstra-like backward relaxation.
    # Our heap stores tuples (-dp[station], station)
    h = []
    heapq.heappush(h, (-dp[N], N))
    
    while h:
        cur_val, v = heapq.heappop(h)
        cur_val = -cur_val
        if cur_val != dp[v]:
            continue
        # Relax all incoming edges (which represent train schedules from some station u to v)
        for u, l, d, k, c in rev[v]:
            # A train from u -> v departs at time t and arrives at t+c.
            # We require that t + c <= dp[v] (dp[v] is the maximum allowed arrival time we can use at v).
            # In the schedule, trains depart at times: l, l+d, l+2d, ..., l+(k-1)*d.
            # Let R = dp[v] - c be the maximum departure time from u (for this train) so that arrival is valid.
            # We need to choose the maximum t = l + d*j that is <= R.
            R = dp[v] - c
            if R < l:
                continue  # no train in the given schedule can satisfy the requirement.
            # Compute maximum j such that l + d*j <= R:
            j = (R - l) // d
            if j >= k:
                j = k - 1
            candidate = l + d * j
            # If this candidate is better than our current best for station u, update dp[u].
            if candidate > dp[u]:
                dp[u] = candidate
                heapq.heappush(h, (-dp[u], u))
    
    # For each station from 1 to N-1, output the dp value if reachable; otherwise print "Unreachable"
    out_lines = []
    for i in range(1, N):
        if dp[i] == NEG:
            out_lines.append("Unreachable")
        else:
            out_lines.append(str(dp[i]))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == "__main__":
    main()