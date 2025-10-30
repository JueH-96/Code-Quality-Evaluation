def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    X = list(map(int, input_data[2:]))

    # A fast way to compute clockwise and ring distances
    def cw_dist(a, b, n):
        # Clockwise distance from a to b on a ring of size n
        # (assuming islands are labeled 1..n).
        # If b >= a: b - a
        # else: (b - a + n)
        if b >= a:
            return b - a
        return b - a + n

    def ring_dist(a, b, n):
        # Shortest distance on the ring
        d = abs(a - b)
        return min(d, n - d)

    # We will do the following:
    # 1) Compute baseDist = sum of the shortest (ring) distances
    #    between consecutive visited islands.
    # 2) For each consecutive pair, let d_k = ringDist(X_k, X_{k+1}),
    #    and W_k = N - 2*d_k.  If W_k > 0, we add W_k to all edges
    #    that lie on the chosen "shorter path" for that pair.
    #
    #    The reason: If an edge e is used by the shorter path of pair k,
    #    and that edge e is removed, we must take the other path of length (N - d_k),
    #    incurring an extra cost of (N - d_k) - d_k = N - 2*d_k = W_k.
    #
    # 3) After we fill in a "weight" E[e] for each edge e (1..N),
    #    the cost if we remove edge e is baseDist + E[e].  We want the minimum of this.
    #
    # To fill E[e] efficiently, we use a difference-array technique on a ring.

    # Step 1: Compute baseDist
    baseDist = 0
    # We'll keep a difference array of length N+2 so we can safely do range updates
    diff = [0]*(N+2)

    for i in range(M-1):
        a = X[i]
        b = X[i+1]
        d = ring_dist(a, b, N)  # the length of the chosen short path 
        baseDist += d
        # W = how much extra cost if the chosen path is blocked
        W = N - 2*d
        if W <= 0:
            # If W=0, blocking that path costs no extra, so skip
            # (distance is N/2 for an even? Actually if d=N/2 for even N, W=0.)
            continue

        # Determine which direction is the chosen shorter path
        # If cw_dist(a,b) <= cw_dist(b,a), we use a->b clockwise
        cw_ab = cw_dist(a, b, N)
        cw_ba = cw_dist(b, a, N)
        if cw_ab <= cw_ba:
            # path is a -> ... -> b (clockwise)
            start = a
            end = (b - 1) if b > 1 else (b - 1 + N)
        else:
            # path is b -> ... -> a (clockwise)
            start = b
            end = (a - 1) if a > 1 else (a - 1 + N)

        # In our edge-indexing scheme: edge i connects island i to i+1 (mod N).
        # So the "set of edges" from start..end (clockwise) is edges
        # start, start+1, ..., end (mod N).
        # We'll do a range-add in circular sense on [start, end].
        if start <= end:
            diff[start] += W
            diff[end+1] -= W
        else:
            # The range wraps around the end of the array
            diff[start] += W
            diff[N+1] -= W  # from start..N
            diff[1] += W    # from 1..end
            diff[end+1] -= W

    # Now build the prefix sums to get E[e] = total W for edge e
    E = [0]*(N+1)
    run = 0
    for i in range(1, N+1):
        run += diff[i]
        E[i] = run

    # We want min_{e=1..N} of E[e], because removing edge e means we add E[e] to the baseDist
    answer = baseDist + min(E[1:])  # E[1..N]
    print(answer)