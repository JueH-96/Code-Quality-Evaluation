def solve():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    info = input_data[2:]

    # Adjacency will be built "backwards" so we can process from station N backwards.
    # For a train A -> B, we'll store an edge from B back to A with (l, d, k, c).
    # That way, knowing the best time we can be at B (G[B]), we can find the best
    # departure time from A that still arrives at B in time G[B].
    adjacency = [[] for _ in range(N+1)]
    idx = 0
    for _ in range(M):
        l_i = int(info[idx]); d_i = int(info[idx+1]); k_i = int(info[idx+2])
        c_i = int(info[idx+3]); A_i = int(info[idx+4]); B_i = int(info[idx+5])
        idx += 6
        # Store an edge: from B_i -> A_i, with parameters (l_i, d_i, k_i, c_i)
        adjacency[B_i].append((A_i, l_i, d_i, k_i, c_i))

    # We define G[i] = the maximum time one can start at station i and still eventually reach station N.
    # Initialize G[N] = "infinity" (a large value) and G[others] = -infinity.
    INF = 10**20  # large enough sentinel for "infinity" in this problem
    NEG_INF = -10**20

    G = [NEG_INF]*(N+1)
    G[N] = INF

    # We'll use a max-heap to propagate updates (Python's heapq is a min-heap, so store negative keys).
    heap = []
    heapq.heappush(heap, (-G[N], N))

    def get_max_departure_time(l, d, k, Tcap):
        """
        Return the largest t in the arithmetic progression l + q*d (0 <= q < k)
        that is <= Tcap. If none is feasible, return NEG_INF.
        """
        if Tcap < l:
            return NEG_INF
        # q = floor((Tcap - l) / d)
        q = (Tcap - l)//d
        if q < 0:
            return NEG_INF
        if q >= k:
            q = k-1
        return l + q*d

    # Standard D'-like loop:
    while heap:
        # current best station has largest G-value
        current_val, station = heapq.heappop(heap)
        current_val = -current_val
        # If this doesn't match the recorded G[station], skip
        if current_val < G[station]:
            continue
        # Relax edges: for each backward edge (station -> prev_station)
        for (prev_st, l_i, d_i, k_i, c_i) in adjacency[station]:
            # We want t + c_i <= G[station], so t <= G[station] - c_i
            if G[station] == INF:
                # Then Tcap is "infinity", so the maximum departure is just
                # the largest in l_i + (k_i-1)*d_i
                candidate_departure = l_i + (k_i-1)*d_i
            else:
                Tcap = G[station] - c_i
                candidate_departure = get_max_departure_time(l_i, d_i, k_i, Tcap)
            if candidate_departure > G[prev_st]:
                G[prev_st] = candidate_departure
                heapq.heappush(heap, (-candidate_departure, prev_st))

    # Now output f(1), f(2), ..., f(N-1).
    # If G[i] is still NEG_INF (or less), print "Unreachable".
    # Otherwise print G[i].
    for s in range(1, N):
        if G[s] < -10**19:  # effectively NEG_INF
            print("Unreachable")
        else:
            print(G[s])