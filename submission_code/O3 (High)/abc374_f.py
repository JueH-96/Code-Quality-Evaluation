import sys


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    N, K, X = map(int, sys.stdin.readline().split())
    T = list(map(int, sys.stdin.readline().split()))  # already non-decreasing

    # prefix sums of T_i for O(1) segment sums
    pref = [0] * (N + 1)
    for i in range(N):
        pref[i + 1] = pref[i] + T[i]

    NEG_INF = -10 ** 18          # sufficiently small sentinel for “no previous shipment”

    # dp[i] : dict { last_shipping_day : minimal dissatisfaction after shipping first i orders }
    dp = [dict() for _ in range(N + 1)]
    dp[0][NEG_INF] = 0           # no order has been shipped yet

    for i in range(N + 1):
        for last_day, cur_cost in list(dp[i].items()):
            # try to ship the next d (1 … K) orders together
            for d in range(1, K + 1):
                j = i + d
                if j > N:
                    break

                latest_release = T[j - 1]                       # last order in the batch
                ship_day = max(latest_release, last_day + X)    # earliest feasible day

                seg_sum = pref[j] - pref[i]
                add_cost = d * ship_day - seg_sum               # dissatisfaction for this batch
                new_cost = cur_cost + add_cost

                # keep the best cost for state (j, ship_day)
                if ship_day not in dp[j] or new_cost < dp[j][ship_day]:
                    dp[j][ship_day] = new_cost

    # answer = minimum cost among all states that have shipped all N orders
    answer = min(dp[N].values())
    print(answer)


if __name__ == "__main__":
    main()