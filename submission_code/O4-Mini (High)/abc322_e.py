import sys
def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    P = int(next(it))
    C = [0]*N
    A = [[0]*K for _ in range(N)]
    for i in range(N):
        C[i] = int(next(it))
        for j in range(K):
            A[i][j] = int(next(it))
    # DP over states of K parameters, each in [0..P]
    base = P + 1
    # precompute base powers
    pow_base = [1]*K
    for j in range(1, K):
        pow_base[j] = pow_base[j-1] * base
    max_states = base**K
    INF = 10**18
    # dp[s] = min cost to reach state s
    dp = [INF] * max_states
    dp[0] = 0
    # iterate over plans
    for i in range(N):
        cost_i = C[i]
        ai = A[i]
        dp2 = dp[:]  # copy current dp
        # try taking plan i from each existing state
        for s in range(max_states):
            if dp[s] == INF:
                continue
            new_cost = dp[s] + cost_i
            # decode s and form next state
            t = s
            new_idx = 0
            for j in range(K):
                cur = t % base
                t //= base
                nxt = cur + ai[j]
                if nxt > P:
                    nxt = P
                new_idx += pow_base[j] * nxt
            # relax
            if dp2[new_idx] > new_cost:
                dp2[new_idx] = new_cost
        dp = dp2
    # target state: all parameters = P
    target = sum(pow_base[j] * P for j in range(K))
    ans = dp[target]
    if ans >= INF:
        print(-1)
    else:
        print(ans)

main()