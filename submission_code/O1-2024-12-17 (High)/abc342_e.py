def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    it = 0

    N = int(input_data[it]); it += 1
    M = int(input_data[it]); it += 1

    # Build reverse adjacency: for each edge A->B, store it in rev_adj[B]
    # as (A, l, d, k, c).
    rev_adj = [[] for _ in range(N+1)]
    for _ in range(M):
        l_ = int(input_data[it]); it += 1
        d_ = int(input_data[it]); it += 1
        k_ = int(input_data[it]); it += 1
        c_ = int(input_data[it]); it += 1
        A_ = int(input_data[it]); it += 1
        B_ = int(input_data[it]); it += 1
        rev_adj[B_].append((A_, l_, d_, k_, c_))

    # We will compute f(s) = the latest time one can start at station s
    # and still eventually reach station N.  Use a max-heap (priority queue)
    # working backwards from station N.

    NEG_INF = -1         # signifies no route (f(s) = -∞ in the statement)
    INF = 10**20         # a large "infinity" sentinel
    f = [NEG_INF] * (N+1)
    f[N] = INF           # from station N, we consider arrival time unbounded

    # Max-heap of (time, station). Python has min-heap, so store negative times.
    pq = []
    heapq.heappush(pq, (-f[N], N))

    def get_departure_time(arrival_limit, l, d, k, c):
        """
        Given that we must arrive by `arrival_limit` at some station,
        and an edge has trains departing at times l + n*d (0 <= n < k),
        each taking c time to travel,
        return the maximum feasible departure time T' (or NEG_INF if none)
        such that T' + c <= arrival_limit.
        """
        if arrival_limit == NEG_INF:
            return NEG_INF
        X = arrival_limit - c
        if X < l:
            return NEG_INF
        # We want T' = max { l + n*d | 0 <= n < k } subject to l + n*d <= X.
        tmp = X - l
        nmax = tmp // d
        if nmax < 0:
            return NEG_INF
        if nmax >= k:
            nmax = k - 1
        return l + nmax*d

    # D' - style loop (in reverse). Relax edges in descending order of f-values
    while pq:
        cur_neg, station = heapq.heappop(pq)
        cur_val = -cur_neg
        # If this is out of date, skip
        if cur_val < f[station]:
            continue

        # Try to improve predecessors (u -> station)
        for (u, l_, d_, k_, c_) in rev_adj[station]:
            dep_time = get_departure_time(cur_val, l_, d_, k_, c_)
            if dep_time > f[u]:
                f[u] = dep_time
                heapq.heappush(pq, (-dep_time, u))

    # Output results for stations 1..N-1
    # If f(s) == -1, print "Unreachable", else print f(s).
    ans = []
    for s in range(1, N):
        if f[s] == NEG_INF:
            ans.append("Unreachable")
        else:
            ans.append(str(f[s]))
    print("
".join(ans))

# Do not forget to call main().
if __name__ == "__main__":
    main()