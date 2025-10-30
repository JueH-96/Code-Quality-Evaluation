import sys
import threading

def main():
    import sys
    import bisect

    input = sys.stdin.readline
    N, X = map(int, input().split())
    # items_by_vit[v] will hold (A_i, C_i) for vitamin v=1,2,3
    items_by_vit = {1: [], 2: [], 3: []}
    for _ in range(N):
        v, a, c = map(int, input().split())
        items_by_vit[v].append((a, c))

    # For each vitamin type v, compute dp_v[c] = maximum total A achievable
    # with exactly c calories, then take prefix max so dp_v is non-decreasing in c.
    dp = {}
    for v in (1, 2, 3):
        # initialize dp_v
        dp_v = [0] * (X + 1)
        for a, c in items_by_vit[v]:
            # 0/1 knapsack step
            # traverse calories from high to low
            for cal in range(X, c - 1, -1):
                # either skip this item or take it
                val = dp_v[cal - c] + a
                if val > dp_v[cal]:
                    dp_v[cal] = val
        # make prefix maximum so dp_v is the best for "at most" cal
        for cal in range(1, X + 1):
            if dp_v[cal - 1] > dp_v[cal]:
                dp_v[cal] = dp_v[cal - 1]
        dp[v] = dp_v

    # The maximum T possible is limited by the smallest total sum we can get
    maxT = min(dp[1][X], dp[2][X], dp[3][X])

    # helper to check if we can achieve at least T of each vitamin within X calories
    def can_do(T):
        # if T=0 always ok
        if T == 0:
            return True
        total_cal = 0
        for v in (1, 2, 3):
            dp_v = dp[v]
            # if even with all calories we can't reach T, fail
            if dp_v[X] < T:
                return False
            # binary search for smallest cal s.t. dp_v[cal] >= T
            # dp_v is non-decreasing
            idx = bisect.bisect_left(dp_v, T)
            total_cal += idx
            if total_cal > X:
                return False
        return total_cal <= X

    # binary search on T in [0..maxT]
    lo, hi = 0, maxT
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if can_do(mid):
            lo = mid
        else:
            hi = mid - 1

    print(lo)

if __name__ == "__main__":
    main()