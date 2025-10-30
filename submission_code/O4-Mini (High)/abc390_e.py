import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    X = int(next(it))
    items = {1: [], 2: [], 3: []}
    for _ in range(N):
        v = int(next(it))
        a = int(next(it))
        c = int(next(it))
        items[v].append((c, a))
    # Build dp_j for each vitamin type j: dp_j[w] = max total A with cost w
    NEG_INF = -10**18
    dp = {}
    for j in (1, 2, 3):
        dpj = [NEG_INF] * (X + 1)
        dpj[0] = 0
        for c, a in items[j]:
            # 0/1 knapsack on cost dimension
            for w in range(X, c - 1, -1):
                val = dpj[w - c] + a
                if val > dpj[w]:
                    dpj[w] = val
        dp[j] = dpj
    # maximum possible t is limited by smallest dp_j[X]
    t_max = min(dp[1][X], dp[2][X], dp[3][X])
    if t_max < 0:
        # if some type cannot get even 0, answer is 0
        print(0)
        return
    # function to check if we can achieve min intake >= t within calorie X
    def feasible(t):
        total_cost = 0
        # for each type, find minimal cost w such that dp_j[w] >= t
        for j in (1, 2, 3):
            dpj = dp[j]
            # if even at max cost we can't reach t, fail
            if dpj[X] < t:
                return False
            # find minimal w
            # linear scan from 0 to X
            # (dpj[0] >= 0 always for t=0, so loop always finds for t=0)
            wj = None
            for w in range(0, X + 1):
                if dpj[w] >= t:
                    wj = w
                    break
            # wj must exist
            total_cost += wj
            if total_cost > X:
                return False
        return total_cost <= X

    # binary search on t in [0, t_max]
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