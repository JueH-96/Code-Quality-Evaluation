def main():
    import sys
    input = sys.stdin.readline
    from bisect import bisect_left

    N, X = map(int, input().split())
    # Separate items by vitamin type
    groups = {1: [], 2: [], 3: []}
    for _ in range(N):
        v, a, c = map(int, input().split())
        groups[v].append((c, a))

    # For each vitamin group, do a 0/1 knapsack over calories to maximize vitamin sum
    # dp_g[c] = max total vitamin-A achievable with exactly c calories (or up to c, since non-decreasing)
    dp = {}
    for g in (1, 2, 3):
        items = groups[g]
        dpg = [0] * (X + 1)
        for cost, val in items:
            # traverse calories backward
            for cal in range(X, cost - 1, -1):
                newv = dpg[cal - cost] + val
                if newv > dpg[cal]:
                    dpg[cal] = newv
        # make non-decreasing: more calories never reduce max vitamin
        for cal in range(1, X + 1):
            if dpg[cal] < dpg[cal - 1]:
                dpg[cal] = dpg[cal - 1]
        dp[g] = dpg

    # The maximum possible t cannot exceed the minimal of what each group can provide
    t_max = min(dp[1][X], dp[2][X], dp[3][X])

    # A function to check if we can achieve at least t of each vitamin within X calories
    def feasible(t):
        total_cal = 0
        for g in (1, 2, 3):
            dpg = dp[g]
            # find minimal c such that dpg[c] >= t
            idx = bisect_left(dpg, t)
            if idx > X:
                return False
            total_cal += idx
            if total_cal > X:
                return False
        return True

    # Binary search on t in [0..t_max]
    lo, hi = 0, t_max
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid - 1

    print(lo)

if __name__ == "__main__":
    main()