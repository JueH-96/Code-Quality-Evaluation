def solve():
    N = int(input())
    A = list(map(int, input().split()))

    dp = [0] * N
    dp[0] = 1
    dp[1] = 2 if A[1] - A[0] == A[2] - A[1] else 1

    for i in range(2, N):
        if A[i] - A[i-1] == A[i-1] - A[i-2]:
            dp[i] = dp[i-1] + 1
        else:
            dp[i] = 2

    print(sum(dp))

solve()