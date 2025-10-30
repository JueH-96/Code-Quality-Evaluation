import sys
input = sys.stdin.read

def max_currency():
    data = input().split()
    index = 0

    N = int(data[index])
    index += 1
    A = list(map(int, data[index:index + N]))
    index += N
    S = []
    T = []

    for i in range(N - 1):
        S.append(int(data[index]))
        T.append(int(data[index + 1]))
        index += 2

    # Dynamic Programming to find the maximum possible currency of country N
    dp = [0] * N
    dp[0] = A[0]

    for i in range(1, N):
        max_val = 0
        for j in range(i):
            if S[j] <= dp[j]:
                max_val = max(max_val, (dp[j] // S[j]) * T[j])
        dp[i] = A[i] + max_val

    print(dp[N - 1])

max_currency()