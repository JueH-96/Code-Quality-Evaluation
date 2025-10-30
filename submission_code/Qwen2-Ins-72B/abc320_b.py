S = input()
N = len(S)
dp = [[False] * N for _ in range(N)]
max_len = 1
for i in range(N):
    dp[i][i] = True
    if i < N - 1 and S[i] == S[i + 1]:
        dp[i][i + 1] = True
        max_len = 2
for length in range(3, N + 1):
    for i in range(N - length + 1):
        j = i + length - 1
        if S[i] == S[j] and dp[i + 1][j - 1]:
            dp[i][j] = True
            max_len = length
print(max_len)