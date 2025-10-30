T = int(input())
for _ in range(T):
    N, X, K = map(int, input().split())
    ans = 0
    while X <= N and K >= 0:
        if K == 0:
            ans += 1
        X *= 2
        K -= 1
    print(ans)