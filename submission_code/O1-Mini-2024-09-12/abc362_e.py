def solve():
    import sys
    from collections import defaultdict

    mod = 998244353
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))

    res = [0] * (N + 1)
    res[1] = N

    dp = [defaultdict(lambda: [0] * (N + 2)) for _ in range(N)]

    # Initialize for k=2
    for j in range(N):
        for i in range(j):
            d = A[j] - A[i]
            dp[j][d][2] = (dp[j][d][2] + 1) % mod
            res[2] = (res[2] + 1) % mod

    # Fill dp for k >=3
    for k in range(3, N + 1):
        for j in range(N):
            for i in range(j):
                d = A[j] - A[i]
                cnt = dp[i][d][k - 1]
                if cnt:
                    dp[j][d][k] = (dp[j][d][k] + cnt) % mod
                    res[k] = (res[k] + cnt) % mod

    output = ' '.join(str(res[k] % mod) for k in range(1, N + 1))
    print(output)

if __name__ == "__main__":
    solve()