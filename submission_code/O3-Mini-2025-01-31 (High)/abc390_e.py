def main():
    import sys
    import numpy as np
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    X = int(next(it))
    
    # Partition foods by vitamin type 1,2,3.
    groups = {1: [], 2: [], 3: []}
    for i in range(n):
        v = int(next(it))
        a = int(next(it))
        c = int(next(it))
        # If the cost exceeds X, it can never be used.
        if c <= X:
            groups[v].append((c, a))
    
    # We define a function to compute the dp array (using 0/1 knapSack)
    # for each vitamin group. The dp array will be of size (X+1) and
    # dp[c] is the maximum vitamin units that can be achieved with EXACT cost c,
    # or a very low number (NEG_INF) if not reachable.
    #
    # Then we compute best[c] = max(dp[0], dp[1], ..., dp[c]) so that best is
    # non-decreasing.
    def compute_best_for_group(items, capacity):
        # we choose a NEG_INF that is small enough
        NEG_INF = -10**12  
        dp = np.full(capacity+1, NEG_INF, dtype=np.int64)
        dp[0] = 0
        # Process each food item only once (0/1 knapsack)
        for cost, vitamin in items:
            # Make a copy so that the update does not "reuse" this item more than once.
            newdp = dp.copy()
            # For indices from cost to capacity, try taking this item:
            # newdp[j] = max(newdp[j], dp[j - cost] + vitamin).
            # We do this vectorized update:
            cand = dp[:capacity+1 - cost] + vitamin
            newdp[cost:] = np.maximum(newdp[cost:], cand)
            dp = newdp
        # Compute the prefix maximum so that best[c] is the best attainable using at most c calories.
        best = np.maximum.accumulate(dp)
        return best

    # Compute the best array for each vitamin group.
    best_by_group = {}
    for vit in (1, 2, 3):
        best_by_group[vit] = compute_best_for_group(groups[vit], X)
    
    # For each group the maximum vitamin sum achievable (with cost at most X) is best[X].
    max1 = int(best_by_group[1][X])
    max2 = int(best_by_group[2][X])
    max3 = int(best_by_group[3][X])
    # The answer can never exceed the minimum over these three.
    M_upper = min(max1, max2, max3)
    
    # Given a best array (which is non-decreasing) and target M,
    # we want to compute the smallest c (0 <= c <= X) such that best[c] >= M.
    # If it is not possible, we return a value larger than X.
    def minimal_cost_for_M(best_arr, M):
        idx = int(np.searchsorted(best_arr, M, side="left"))
        if idx < len(best_arr) and best_arr[idx] >= M:
            return idx
        else:
            return X + 1  # not feasible
    
    # Now binary search for the maximum M such that there exists a choice for each group
    # with a total cost <= X.
    lo = 0
    hi = M_upper
    ans = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        c1 = minimal_cost_for_M(best_by_group[1], mid)
        c2 = minimal_cost_for_M(best_by_group[2], mid)
        c3 = minimal_cost_for_M(best_by_group[3], mid)
        if c1 + c2 + c3 <= X:
            ans = mid  # candidate mid is achievable
            lo = mid + 1
        else:
            hi = mid - 1
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()