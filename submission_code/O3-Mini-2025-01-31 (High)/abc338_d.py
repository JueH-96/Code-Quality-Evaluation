def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    # Read the M island numbers that must be visited in order.
    X = [int(next(it)) for _ in range(M)]
    
    # The islands form a cycle. When we remove one bridge, the cycle becomes a chain.
    # For a given removed bridge r (r in {1, 2, ..., N})—specifically the bridge connecting island r and island r+1 (with island N+1 taken as 1)—
    # we assign each island a "position" in the chain:
    #    pos_r(u) = u - r - 1, if u > r, and = u + N - r - 1 if u <= r.
    # In the resulting tree, the distance between two islands x and y is exactly:
    #    |pos_r(x) - pos_r(y)|
    # One can check that if both x,y are on the same side of r (i.e. both > r or both <= r), then the distance is |x-y|,
    # and if they are on different sides, the distance becomes N - |x-y|.
    #
    # For any consecutive pair in the tour, i.e. (X_i, X_{i+1}),
    # let A = min(X_i, X_{i+1]) and B = max(X_i, X_{i+1]).
    # Then if the removed edge r falls in the interval [A, B-1] (note A <= r <= B-1),
    # the pair is "cut" by the removed edge and the distance becomes
    #    N - (B - A)
    # Otherwise, the distance is (B - A).
    #
    # Thus the contribution of the pair (X_i, X_{i+1}) to the tour length when bridge r is removed is:
    #    f_r(X_i, X_{i+1]) = (B - A)      if r is NOT in [A, B-1]
    #                         N - (B - A)  if r is in [A, B-1].
    # We can write this as:
    #    f_r = (B-A)  +  (if r in [A, B-1] then (N - 2*(B-A)) else 0).
    #
    # Let base = sum_{i=1}^{M-1} (B - A) over all consecutive pairs.
    # Then for every r, the overall tour cost is:
    #    F(r) = base + sum_{i, with r in [A_i, B_i-1]} (N - 2*(B_i - A_i)).
    #
    # Our goal is to choose the r (i.e. the bridge to remove) that minimizes F(r).
    #
    # To do this efficiently, we use a difference array (deltaArr) over r = 1,2,..., N:
    #   For each consecutive pair with A and B, we add (N - 2*(B-A)) to all indices r in the interval [A, B-1].
    
    base = 0
    # Create an array for interval updates. We use 1-indexing for r=1,...,N.
    deltaArr = [0] * (N + 2)
    
    for i in range(M - 1):
        a = X[i]
        b = X[i + 1]
        if a < b:
            A = a
            B = b
        else:
            A = b
            B = a
        d = B - A
        base += d
        # For all r in the interval [A, B-1], add delta = N - 2*d.
        # (This is the extra cost incurred when the removed edge lies between A and B.)
        if A <= B - 1:  # Always true since X_i != X_{i+1} so d>=1.
            delta = N - 2 * d
            deltaArr[A] += delta
            # We subtract at B (since the interval is [A, B-1]).
            deltaArr[B] -= delta
    
    # Now, for each possible r from 1 to N, compute the total extra cost (via prefix sum)
    # and hence the tour cost F(r) = base + (extra cost).
    best = None
    current = 0
    for r in range(1, N + 1):
        current += deltaArr[r]
        cost = base + current
        if best is None or cost < best:
            best = cost
    
    sys.stdout.write(str(best))

if __name__ == '__main__':
    main()