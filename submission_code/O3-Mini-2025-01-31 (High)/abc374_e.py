def main():
    import sys, math
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    X = int(next(it))
    processes = []
    for _ in range(N):
        A = int(next(it)); P = int(next(it))
        B = int(next(it)); Q = int(next(it))
        processes.append((A, P, B, Q))
    
    # For a given process with parameters A, P, B, Q,
    # we want to achieve at least k products per day.
    # We can buy x units of machine S (capacity A; cost P)
    # and y units of machine T (capacity B; cost Q)
    # so that x*A + y*B >= k.
    # Our goal is to compute the minimum cost for this process.
    # A useful observation for two-coin problems is that
    # one can find an optimum by trying a small set of x values.
    # In particular, one of the optimal solutions always has either
    # x < B or y < A.
    # Thus, it suffices to try x in the union of two ranges:
    #   (i) x from 0 up to min(B, x_max),
    #   (ii) x from max(0, x_max - B) up to x_max,
    # where x_max is the minimal number of S machines needed if used alone.
    
    def min_cost_for_process(A, P, B, Q, k):
        if k <= 0:
            return 0
        # x_max: if we use only S-machines.
        x_max = (k + A - 1) // A
        candidates = set()
        # Try small x values: 0 <= x < min(B, x_max+1)
        for x in range(min(B, x_max + 1)):
            candidates.add(x)
        # Also try values very near x_max.
        for x in range(max(0, x_max - B), x_max + 1):
            candidates.add(x)
        best = 10**18
        for x in candidates:
            prod_from_S = x * A
            rem = k - prod_from_S
            if rem <= 0:
                y = 0
            else:
                y = (rem + B - 1) // B  # Ceil division: need enough T-machines.
            cost_here = x * P + y * Q
            if cost_here < best:
                best = cost_here
        return best

    # Given a target production capacity k (meaning every process produces at least k)
    # check if, using an optimal mix for every process, the total cost does not exceed X.
    def is_feasible(k):
        total = 0
        for (A, P, B, Q) in processes:
            cost = min_cost_for_process(A, P, B, Q, k)
            total += cost
            if total > X:
                return False
        return total <= X

    # We now perform binary search on k.
    # For k = 0, the cost is 0 (by not purchasing any machines) so it is always feasible.
    # We need an upper bound.
    # A safe upper bound for process i is if we allocated all budget X to it.
    # In continuous relaxation, the best production per yen is max(A/P, B/Q).
    # So process i cannot exceed roughly X * max(A/P, B/Q) in production.
    # Hence, an overall upper bound is:
    hi_candidates = []
    for (A, P, B, Q) in processes:
        ratio = max(A / P, B / Q)
        hi_candidates.append(int(X * ratio) + 1)
    hi_bound = min(hi_candidates)

    lo = 0
    hi = hi_bound
    ans = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if is_feasible(mid):
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1

    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()