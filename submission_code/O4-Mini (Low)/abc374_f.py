def main():
    import sys
    input = sys.stdin.readline
    N, K, X = map(int, input().split())
    T = list(map(int, input().split()))
    # prefix sums of T
    ps = [0] * (N + 1)
    for i in range(N):
        ps[i+1] = ps[i] + T[i]
    # dp[i]: list of states (last_ship_day, cost) for first i orders
    dp = [[] for _ in range(N+1)]
    # before any shipment, pretend last shipment was at day -X, cost 0
    dp[0] = [(-X, 0)]
    for i in range(N):
        if not dp[i]:
            continue
        # try batches of size b from this state
        for last_s, cost in dp[i]:
            # next batch size
            for b in range(1, K+1):
                j = i + b
                if j > N:
                    break
                # the latest order time in this batch is T[j-1]
                base = T[j-1]
                # we must wait until at least last_s + X
                S = base if base >= last_s + X else last_s + X
                # cost add = b*S - sum(T[i..j-1])
                add = b * S - (ps[j] - ps[i])
                dp[j].append((S, cost + add))
        # prune dp[i+1..i+K] lazily after all additions? 
        # But simpler: after finishing all states for i, prune dp[i+1..i+K]
        # Actually better to prune each dp[j] here.
        for j in range(i+1, min(N, i+K) + 1):
            states = dp[j]
            if not states:
                continue
            # sort by last_s increasing, then cost increasing
            states.sort(key=lambda x: (x[0], x[1]))
            pruned = []
            best_cost = 10**40
            # for increasing last_s, keep only if cost < best_cost
            for s, c in states:
                if c < best_cost:
                    pruned.append((s, c))
                    best_cost = c
            dp[j] = pruned
    # answer is minimal cost among dp[N]
    ans = min(c for _, c in dp[N]) if dp[N] else 0
    print(ans)

if __name__ == "__main__":
    main()