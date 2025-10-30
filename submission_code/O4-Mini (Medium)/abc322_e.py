import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    P = int(next(it))

    plans = []
    for _ in range(N):
        c = int(next(it))
        a = [int(next(it)) for __ in range(K)]
        plans.append((c, a))

    # Use a DP dictionary: state (tuple of K values) -> min cost
    INF = 10**30
    dp = { (0,)*K : 0 }

    for c, a in plans:
        # newdp starts as a copy of dp (we can skip this plan)
        newdp = dict(dp)
        # try taking this plan from every existing state
        for state, cost_so_far in dp.items():
            new_state = list(state)
            for j in range(K):
                # increase and cap at P
                v = new_state[j] + a[j]
                if v > P:
                    v = P
                new_state[j] = v
            new_state = tuple(new_state)
            new_cost = cost_so_far + c
            prev = newdp.get(new_state, INF)
            if new_cost < prev:
                newdp[new_state] = new_cost
        dp = newdp

    target = (P,)*K
    ans = dp.get(target, INF)
    if ans >= INF:
        print(-1)
    else:
        print(ans)

if __name__ == "__main__":
    main()