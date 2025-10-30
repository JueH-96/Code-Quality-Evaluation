from collections import defaultdict

MOD = 998244353

def solve():
    N, Q = map(int, input().split())
    P = []
    V = []
    for _ in range(Q):
        p, v = map(int, input().split())
        P.append(p)
        V.append(v)

    dp = defaultdict(int)
    dp[tuple([0] * N)] = 1

    for i in range(Q):
        new_dp = defaultdict(int)
        for state in dp:
            state = list(state)
            for j in range(P[i] - 1, -1, -1):
                if state[j] > V[i]:
                    break
                state[j] = V[i]
            state = tuple(state)
            new_dp[state] += dp[tuple(state)]
            new_dp[state] %= MOD

            for j in range(P[i] - 1, N):
                if state[j] > V[i]:
                    break
                state[j] = V[i]
            state = tuple(state)
            new_dp[state] += dp[tuple(state)]
            new_dp[state] %= MOD

        dp = new_dp

    return sum(dp.values()) % MOD

print(solve())