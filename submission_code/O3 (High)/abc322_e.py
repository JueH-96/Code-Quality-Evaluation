import sys


def main() -> None:
    # Read input
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    P = int(next(it))

    plans = []
    for _ in range(N):
        c = int(next(it))
        vec = [int(next(it)) for _ in range(K)]
        plans.append((c, vec))

    INF = 10 ** 19  # bigger than any possible cost
    # dp maps a state tuple to its minimal cost
    # initial state: all parameters are zero at cost 0
    dp = {(0,) * K: 0}

    for cost, add in plans:
        new_dp = dp.copy()  # copy because each plan can be used at most once
        for state, cur_cost in dp.items():
            # compute the state after taking this plan
            nxt = tuple(min(P, state[i] + add[i]) for i in range(K))
            ncost = cur_cost + cost
            if ncost < new_dp.get(nxt, INF):
                new_dp[nxt] = ncost
        dp = new_dp

    target = (P,) * K
    if target in dp:
        print(dp[target])
    else:
        print(-1)


if __name__ == "__main__":
    main()