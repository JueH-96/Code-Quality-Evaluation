def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    idx = 2

    # We will build a reverse adjacency list:
    # For each station b, we keep a list of (a, l, d, k, c)
    # meaning: there's a train from a to b departing at times l + n*d (0 <= n < k),
    # each arriving at time (l + n*d) + c.
    # So if we can be at b at time T, we can try to find a departure time from a
    # that arrives at b by time T, i.e. departure_time + c <= T.
    # Then departure_time <= T - c, and departure_time must lie in the arithmetic
    # progression [l, l + (k-1)*d].
    # We'll do a backward D' approach with a max-heap (by time).

    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        l_i = int(input_data[idx]); idx+=1
        d_i = int(input_data[idx]); idx+=1
        k_i = int(input_data[idx]); idx+=1
        c_i = int(input_data[idx]); idx+=1
        A_i = int(input_data[idx]); idx+=1
        B_i = int(input_data[idx]); idx+=1
        adj[B_i].append((A_i, l_i, d_i, k_i, c_i))

    # f[s] = the maximum time one can start at station s and still reach station N.
    # Initialize f[N] = "infinity" (a very large number), others -1 (meaning unreachable so far).
    INF = 10**20
    f = [-1]*(N+1)
    f[N] = INF

    # Use a max-heap (store negative times for Python's min-heap) 
    pq = []
    # Start from station N with time = INF
    heapq.heappush(pq, (-f[N], N))

    while pq:
        negT, st = heapq.heappop(pq)
        T = -negT
        # If this popped time is stale, ignore
        if T < f[st]:
            continue
        # Otherwise, try to relax neighbors in reverse
        for (a, l_i, d_i, k_i, c_i) in adj[st]:
            limit = T - c_i  # we must depart by this time
            if limit < l_i:
                # can't depart at or after the earliest departure if limit < l_i
                continue
            # number of steps in arithmetic progression
            # x = floor( (limit - l_i)/d_i ), but also x < k_i
            x = (limit - l_i)//d_i
            if x >= k_i:
                x = k_i - 1
            if x < 0:
                continue
            candidate_departure = l_i + x*d_i
            if candidate_departure > f[a]:
                f[a] = candidate_departure
                heapq.heappush(pq, (-candidate_departure, a))

    # Output f(1), f(2), ..., f(N-1)
    # If f(i) < 0 => Unreachable
    out = []
    for i in range(1, N):
        if f[i] < 0:
            out.append("Unreachable")
        else:
            out.append(str(f[i]))
    print("
".join(out))

# Do not forget to call main()
if __name__ == "__main__":
    main()