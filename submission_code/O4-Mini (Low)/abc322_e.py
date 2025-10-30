import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    INF = 10**30

    # Read input
    N, K, P = map(int, input().split())
    plans = []
    for _ in range(N):
        data = list(map(int, input().split()))
        C = data[0]
        A = data[1:]
        plans.append((C, A))

    # Number of states in each dimension = P+1
    dim = [P+1] * K
    # Total number of states = (P+1)^K
    total = 1
    for d in dim:
        total *= d

    # Precompute multipliers to flatten/unflatten
    # idx = sum(coord[i] * mult[i])
    mult = [1] * K
    for i in range(1, K):
        mult[i] = mult[i-1] * dim[i-1]

    # Precompute coordinates for every flat index
    coords = [None] * total
    for idx in range(total):
        tmp = idx
        c = [0]*K
        for i in range(K):
            c[i] = tmp % dim[i]
            tmp //= dim[i]
        coords[idx] = c

    # dp[idx] = minimum cost to reach the state corresponding to flat index idx
    dp = [INF] * total
    # initial state all zeros has cost 0
    dp[0] = 0

    # Process each plan (0-1 knapsack style over these states)
    for cost, adds in plans:
        old_dp = dp[:]  # snapshot before applying this plan
        # Try applying this plan from every reachable state
        for idx in range(total):
            cur_cost = old_dp[idx]
            if cur_cost == INF:
                continue
            # compute new coordinates
            c = coords[idx]
            new_idx = 0
            for i in range(K):
                # raise at most to P
                nc = c[i] + adds[i]
                if nc > P:
                    nc = P
                new_idx += nc * mult[i]
            # relax dp
            new_cost = cur_cost + cost
            if new_cost < dp[new_idx]:
                dp[new_idx] = new_cost

    # the target state is (P, P, ..., P)
    target_idx = 0
    for i in range(K):
        target_idx += P * mult[i]

    ans = dp[target_idx]
    if ans >= INF:
        print(-1)
    else:
        print(ans)

if __name__ == "__main__":
    main()