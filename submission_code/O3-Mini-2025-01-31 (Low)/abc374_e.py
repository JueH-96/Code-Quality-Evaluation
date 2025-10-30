def main():
    import sys, math
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    X = int(next(it))
    processes = []
    for _ in range(n):
        A = int(next(it))
        P = int(next(it))
        B = int(next(it))
        Q = int(next(it))
        processes.append((A, P, B, Q))
    
    # Given a production demand "r" for a process with two machine options,
    # we need the minimum cost such that x*A + y*B >= r.
    # We use the observation that an optimum combination can be found by only
    # trying a bounded number of candidates: around the division r // A (or r // B).
    def min_cost_for_process(r, A, P, B, Q):
        if r <= 0:
            return 0
        ans = 10**18
        # We'll try candidates for machine count x near r//A.
        x_base = r // A
        # Candidate range: from max(0, x_base - B - 1) to x_base + B + 1.
        low_x = max(0, x_base - B - 1)
        high_x = x_base + B + 1
        for x in range(low_x, high_x + 1):
            rem = r - A * x
            if rem <= 0:
                y = 0
            else:
                y = (rem + B - 1) // B  # ceiling division
            cost = x * P + y * Q
            if cost < ans:
                ans = cost
        
        # Similarly, try candidates using machine T in primary role.
        y_base = r // B
        low_y = max(0, y_base - A - 1)
        high_y = y_base + A + 1
        for y in range(low_y, high_y + 1):
            rem = r - B * y
            if rem <= 0:
                x = 0
            else:
                x = (rem + A - 1) // A
            cost = x * P + y * Q
            if cost < ans:
                ans = cost
        return ans

    # Check if a production capacity "r" is achievable within the budget X.
    def feasible(r):
        total_cost = 0
        for (A, P, B, Q) in processes:
            c = min_cost_for_process(r, A, P, B, Q)
            total_cost += c
            if total_cost > X:  # early exit if budget exceeded
                return False
        return total_cost <= X

    # Binary search for the maximum production capacity (r) we can achieve.
    # The maximum ratio may be as high as X * max(machine production per cost)
    # Since A_i, B_i <= 100 and cost is at least 1 yen, a safe upper bound is X * 100.
    lo = 0
    hi = X * 100
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    sys.stdout.write(str(best))

if __name__ == '__main__':
    main()