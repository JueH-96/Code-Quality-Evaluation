def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    X = int(next(it))
    procs = []
    for _ in range(n):
        a = int(next(it))
        p = int(next(it))
        b = int(next(it))
        q = int(next(it))
        procs.append((a, p, b, q))
    
    # Helper function: Given parameters for a process and a target capacity "cap",
    # compute the minimum cost (in yen) needed to reach at least "cap" production.
    #
    # For process with:
    #   - Machine S offering a products per unit at cost p,
    #   - Machine T offering b products per unit at cost q,
    # we want nonnegative integers x and y such that:
    #     a*x + b*y >= cap
    # with cost = p*x + q*y minimized.
    #
    # The strategy is to consider using only one type of machine (which gives an obvious candidate)
    # and then to try a “neighborhood search” over one of the variables.
    #
    # Since a and b are at most 100, the optimum must lie within a small window around the extreme values.
    def min_cost_for_process(cap, a, p, b, q):
        if cap <= 0:
            return 0
        
        # simple helper for ceiling division
        def ceil_div(x, y):
            return (x + y - 1) // y

        # Candidate 1: Using only machine S.
        best = p * ceil_div(cap, a)
        # Candidate 2: Using only machine T.
        cand = q * ceil_div(cap, b)
        if cand < best:
            best = cand

        # Now try mixing the two machines.
        #
        # Notice that if we set x (the number of S machines), then we require:
        #      a*x + b*y >= cap  =>  y >= ceil((cap - a*x) / b) if (cap - a*x) > 0.
        #
        # It turns out (by standard techniques in linear optimization) that an optimum
        # can be found by checking x in a small interval near the pure-S solution.
        # Let x0 = ceil(cap / a). Then we search for x in the range [max(0, x0 - b - 2), x0 + b + 2].
        x0 = ceil_div(cap, a)
        low_x = max(0, x0 - b - 2)
        high_x = x0 + b + 2
        for x in range(low_x, high_x + 1):
            rem = cap - a * x
            if rem <= 0:
                y = 0
            else:
                y = ceil_div(rem, b)
            cost = p * x + q * y
            if cost < best:
                best = cost

        # Similarly, search by iterating on y in a neighborhood near the pure-T solution.
        y0 = ceil_div(cap, b)
        low_y = max(0, y0 - a - 2)
        high_y = y0 + a + 2
        for y in range(low_y, high_y + 1):
            rem = cap - b * y
            if rem <= 0:
                x = 0
            else:
                x = ceil_div(rem, a)
            cost = q * y + p * x
            if cost < best:
                best = cost

        return best

    # Given a target production capacity "cap", check whether it is possible to guarantee
    # that every process achieves at least "cap" products per day with total cost <= X.
    def feasible(cap):
        total_cost = 0
        for (a, p, b, q) in procs:
            total_cost += min_cost_for_process(cap, a, p, b, q)
            if total_cost > X:
                return False
        return total_cost <= X

    #
    # Binary search to find the maximum capacity C that is achievable.
    #
    # For each process, if we used all the money X on that process alone,
    # we can get at most (machine production * floor(X / cost)).
    # A safe upper bound for the answer is the minimum over processes of:
    #    max( a * (X // p), b * (X // q) )
    hi = None
    for a, p, b, q in procs:
        proc_max = max(a * (X // p), b * (X // q))
        if hi is None or proc_max < hi:
            hi = proc_max
    if hi is None:
        hi = 0

    lo = 0
    ans = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1

    sys.stdout.write(str(ans))


if __name__ == '__main__':
    main()