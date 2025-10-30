def main():
    import sys
    input = sys.stdin.readline

    N, W = map(int, input().split())
    wv = [tuple(map(int, input().split())) for _ in range(N)]

    # dp[x] will hold the maximum "happiness" achievable with total weight exactly x
    # We'll use a large negative number to represent "impossible".
    NEG_INF = -10**15
    dp = [NEG_INF]*(W+1)
    dp[0] = 0

    for w, v in wv:
        # The maximum number of items we might ever pick from this type is
        # limited by (a) bag capacity W//w, and (b) picking more than v
        # items yields 0 or negative net contribution (k*v - k^2 <= 0 if k >= v).
        K = min(W // w, v)

        # We'll build a new dp array after including this type.
        new_dp = [NEG_INF]*(W+1)

        for current_weight in range(W+1):
            old_val = dp[current_weight]
            if old_val == NEG_INF:
                continue

            # Option k = 0 (pick no items of this type).
            if old_val > new_dp[current_weight]:
                new_dp[current_weight] = old_val

            # Try picking k = 1..K items of this type (if it improves happiness).
            # cost = k * w, value_gain = k*v - k^2
            for k in range(1, K+1):
                cost = k * w
                nxt = current_weight + cost
                if nxt > W:
                    break
                val_gain = k*v - k*k
                if val_gain <= 0:
                    # Once val_gain <= 0, larger k will not help (it only goes down),
                    # so we can break early.
                    break
                candidate = old_val + val_gain
                if candidate > new_dp[nxt]:
                    new_dp[nxt] = candidate

        dp = new_dp

    print(max(dp))

# Call main() to run the solution.
if __name__ == "__main__":
    main()