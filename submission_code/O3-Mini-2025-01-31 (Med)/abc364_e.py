def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    X = int(next(it))
    Y = int(next(it))
    dishes = []
    for _ in range(n):
        a = int(next(it))
        b = int(next(it))
        dishes.append((a, b))
    
    # Only dishes that are "safe" to eat (when taken as a single dish)
    # can be part of the non-overflow chain.
    valid = []
    for a, b in dishes:
        if a <= X and b <= Y:
            valid.append((a, b))
            
    # We want to choose as many dishes as possible from the valid ones
    # so that the total sweetness and saltiness do not exceed X and Y,
    # respectively.
    #
    # (Claim: if such a subset of safe dishes has totals <= (X, Y)
    # then it is always possible to order them so that every prefix (when greedily arranged)
    # stays under the limits.
    # You may check on small examples – see the explanation below.)
    #
    # Use dictionary based 0/1 knapsack DP:
    # dp[(s, t)] = maximum number of dishes chosen (from valid) that sum to (s, t).
    dp = {(0, 0): 0}
    for (a, b) in valid:
        # iterate over a snapshot to avoid using one dish more than once
        items = list(dp.items())
        for (s, t), cnt in items:
            ns = s + a
            nt = t + b
            if ns <= X and nt <= Y:
                newcnt = cnt + 1
                key = (ns, nt)
                if key not in dp or dp[key] < newcnt:
                    dp[key] = newcnt
        # A simple pruning: for each count value, keep only states that are not dominated.
        newdp = {}
        bucket = {}
        for (s, t), cnt in dp.items():
            bucket.setdefault(cnt, []).append((s, t))
        for cnt, lst in bucket.items():
            lst.sort()  # sort by sweetness then saltiness
            best = []
            for s, t in lst:
                # We only keep (s,t) if it is not dominated in saltiness by an earlier state.
                if not best or t < best[-1][1]:
                    best.append((s, t))
            for s, t in best:
                newdp[(s, t)] = cnt
        dp = newdp
    
    r0 = 0  # maximum count of non-overflow dishes (safe chain)
    for v in dp.values():
        if v > r0:
            r0 = v
            
    # It is always possible to “append” one extra dish (which may be unsafe)
    # provided not all dishes were used in the safe chain.
    # Thus if r0 < n then answer = r0 + 1,
    # otherwise (if r0 == n) answer = n.
    if r0 < n:
        ans = r0 + 1
    else:
        ans = r0
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()