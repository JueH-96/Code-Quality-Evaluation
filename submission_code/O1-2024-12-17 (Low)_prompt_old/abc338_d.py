def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    X = list(map(int, input_data[2:]))

    # Helper to compute the circular distance (mod N) in the clockwise direction
    # from a to b (0-based).
    def cw_dist(a, b):
        return (b - a) % N

    # We will compute:
    # 1) baseCost = sum of the shortest distances between consecutive X_k, X_{k+1}.
    # 2) For each consecutive pair, determine which edges on the ring's short path
    #    it uses (either clockwise or counterclockwise).  We store an increment
    #    p = N - 2*shortDist for that arc in a difference array.
    # 3) After building and prefix-summing that array, coverSum[e] tells us how
    #    much extra cost is added if edge e is removed (sum of p for all pairs
    #    that would be forced to take the longer path).
    # 4) The answer is min over e of (baseCost + coverSum[e]).

    # Step 1: compute baseCost, initialize difference array
    baseCost = 0
    diff = [0]*(N+1)  # difference array for edges 0..N-1, plus one extra slot

    # A small helper to add "p" to all edges in the arc [s, t) in the clockwise sense
    def add_arc(s, t, p):
        # We assume 0 <= s, t < N. The arc covers edges s, s+1, ..., (t-1) mod N
        if p == 0:
            return
        if s < t:
            diff[s] += p
            diff[t] -= p
        else:
            # arc wraps around the end
            diff[s] += p
            diff[N] -= p  # this will apply from s..N-1
            diff[0] += p  # apply from 0..
            diff[t] -= p  # stop at t-1
        # We'll prefix-sum later

    # Process each consecutive pair
    for i in range(M-1):
        x = X[i] - 1  # 0-based
        y = X[i+1] - 1
        dist_cw = cw_dist(x, y)  # clockwise distance
        dist_ccw = N - dist_cw   # counterclockwise distance
        short_dist = min(dist_cw, dist_ccw)
        baseCost += short_dist

        # If there's a tie (dist_cw == dist_ccw == N/2), p = 0 => removing any edge won't matter.
        p = N - 2*short_dist  # how much extra cost if the short path is blocked
        if p == 0:
            continue

        # If clockwise is indeed <= counterclockwise, short path is x->y cw
        # otherwise short path is y->x cw.
        if dist_cw <= dist_ccw:
            add_arc(x, y, p)
        else:
            add_arc(y, x, p)

    # Now build coverSum via prefix sum of diff
    coverSum = [0]*N
    running = 0
    for i in range(N):
        running += diff[i]
        coverSum[i] = running

    # The cost if we remove edge i is baseCost + coverSum[i].
    # We want the minimum of these.
    ans = min(baseCost + coverSum[i] for i in range(N))
    print(ans)