def main():
    import sys, math
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    try:
        n = int(next(it))
    except StopIteration:
        return
    pts = []
    for i in range(n):
        x = float(next(it))
        y = float(next(it))
        pts.append((x, y))
        
    # Compute distance if all checkpoints are visited (the “all‐visited” route)
    total_all = 0.0
    for i in range(n-1):
        dx = pts[i+1][0] - pts[i][0]
        dy = pts[i+1][1] - pts[i][1]
        total_all += math.hypot(dx, dy)
        
    # We need only consider skip counts s for which the penalty 2^(s-1) is not huge.
    # In fact, if 2^(s-1) > total_all (the “all‐visited” travel distance) then that skip–count cannot be optimal.
    # Also note that at most you can skip n-2 checkpoints.
    if total_all <= 0:
        smax = 0
    else:
        if total_all < 1:
            smax = 0
        else:
            smax = int(math.floor(math.log(total_all, 2))) + 1
            if smax > n - 2:
                smax = n - 2

    # dp[i][s] = minimum travel distance to reach checkpoint i (0-indexed) while having skipped exactly s checkpoints.
    # There are (smax+1) possible s–values (from 0 to smax).
    INF = float('inf')
    dp = [[INF]*(smax+1) for _ in range(n)]
    dp[0][0] = 0.0

    # Precompute for each checkpoint i, for jump lengths L = 1,...,min(smax+1, n-i),
    # the Euclidean distance from checkpoint i to checkpoint (i+L).
    distFrom = [None]*n
    for i in range(n):
        Lmax = smax + 1
        if Lmax > n - i:
            Lmax = n - i
        dlist = [0.0]*(Lmax+1)  # We'll use indices 1..Lmax (index 0 is unused)
        (x_i, y_i) = pts[i]
        for L in range(1, Lmax+1):
            j = i + L
            dx = pts[j][0] - x_i
            dy = pts[j][1] - y_i
            dlist[L] = math.hypot(dx, dy)
        distFrom[i] = dlist

    # DP: for each state (checkpoint i, current skip count s) try jumps to later checkpoints.
    for i in range(n):
        dpi = dp[i]
        maxL_global = n - i   # j = i+L must be within bounds
        for s in range(smax+1):
            cur = dpi[s]
            if cur == INF:
                continue
            # From state (i, s) a jump of length L adds (L-1) to the skip count.
            # We cannot allow s + (L-1) to exceed smax.
            maxL_here = smax - s + 1
            if maxL_here > maxL_global:
                maxL_here = maxL_global
            dlist = distFrom[i]
            for L in range(1, maxL_here+1):
                new_s = s + (L - 1)
                j = i + L
                newval = cur + dlist[L]
                if newval < dp[j][new_s]:
                    dp[j][new_s] = newval

    # The final answer is the minimum over s of (dp[n-1][s] + penalty);
    # note: penalty is 0 if s==0, else penalty = 2^(s-1)
    best = INF
    for s in range(smax+1):
        dcost = dp[n-1][s]
        if dcost == INF:
            continue
        penalty = 0.0 if s == 0 else (2 ** (s - 1))
        tot = dcost + penalty
        if tot < best:
            best = tot
    sys.stdout.write("{:.10f}".format(best))
    
if __name__ == '__main__':
    main()