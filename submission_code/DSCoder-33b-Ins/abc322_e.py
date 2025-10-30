import sys

def solve():
    N, K, P = map(int, sys.stdin.readline().split())
    plans = []
    for _ in range(N):
        plan = list(map(int, sys.stdin.readline().split()))
        plans.append(plan)

    dp = [0] * (P + 1)
    for i in range(N):
        for j in range(P, 0, -1):
            for k in range(K):
                if dp[j] < dp[max(0, j - plans[i][k + 1])] + plans[i][0]:
                    dp[j] = dp[max(0, j - plans[i][k + 1])] + plans[i][0]

    if dp[P] == 0:
        print(-1)
    else:
        print(dp[P])

solve()