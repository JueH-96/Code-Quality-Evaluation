def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    
    # Recall: a rabbit at (x, y) can move diagonally: (x±1, y±1).
    # Thus, in each jump both coordinates change by 1 in absolute value.
    # So, the parity of (x+y) remains invariant.
    # It follows that a jump from A=(x1,y1) to B=(x2,y2) is possible if and only if (x1+y1) % 2 == (x2+y2) %2.
    # When reachable, note also that in k jumps:
    #   |x2 - x1| <= k and |y2 - y1| <= k.
    # Also k must have the same parity as |x2 - x1| (and |y2-y1|).
    # It turns out that the minimal jumps needed is:
    #   k = max(|x2-x1|, |y2-y1|) provided that (x1+y1)%2 == (x2+y2)%2, 
    # and k is not defined (set to 0) if parity differs.
    #
    # A key observation: for two points with dx = |x2-x1| and dy = |y2-y1| (with same parity condition),
    # max{dx, dy} can also be computed using the transformation:
    # let u = x+y, v = x-y. Then:
    #   |u2-u1| = |(x2+y2) - (x1+y1)| and 
    #   |v2-v1| = |(x2-y2) - (x1-y1)|.
    # Notice that:
    #   max(|dx|,|dy|) = (|u2-u1| + |v2-v1|) // 2.
    #
    # Therefore, for each pair (i, j) that are mutually reachable (i.e. in the same parity group), 
    # we have:
    #   dist(P_i, P_j) = (|u_i - u_j| + |v_i - v_j|) // 2.
    #
    # We need to compute the sum:
    #    Σ(i<j) dist(P_i, P_j)
    #   = 1/2 [ Σ(i<j) |u_i-u_j| + Σ(i<j) |v_i-v_j| ].
    #
    # This summation over all pairs is a standard one; it can be computed in O(m log m) if we sort.
    # In particular, if we sort an array arr = [a0, a1, ..., a_{m-1}],
    # then: Σ(i<j) |a_j - a_i| = Σ_{i=0}^{m-1} a_i * (2*i - m + 1).
    #
    # We must compute this sum for two separate groups:
    # one for points with (x+y) even and one for (x+y) odd.
    
    groups = {0: [], 1: []}
    for _ in range(n):
        x = int(next(it))
        y = int(next(it))
        parity = (x + y) & 1
        # Compute u and v for transformation
        u = x + y
        v = x - y
        groups[parity].append((u, v))
    
    total = 0
    # Process each group separately.
    for parity in (0, 1):
        pts = groups[parity]
        m = len(pts)
        if m < 2:
            continue
        # Separate u and v values.
        u_values = sorted(u for u, v in pts)
        v_values = sorted(v for u, v in pts)
        sum_u = 0
        sum_v = 0
        # Compute sum_{i < j} |u_j - u_i| using formula:
        #   Σ_{i=0}^{m-1} u_i * (2*i - m + 1).
        for i, val in enumerate(u_values):
            sum_u += val * (2 * i - m + 1)
        for i, val in enumerate(v_values):
            sum_v += val * (2 * i - m + 1)
        # For the group the contribution is:
        #   (sum_u + sum_v) // 2
        total += (sum_u + sum_v) // 2
    
    sys.stdout.write(str(total))
    
if __name__ == "__main__":
    main()