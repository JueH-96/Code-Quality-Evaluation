N = int(input())
A = []
S = []
for i in range(N):
    a, s = input().split()
    A.append(int(a))
    S.append(s)

dp = [[float('inf')]*4 for _ in range(N+1)]
dp[0][0] = 0
dp[0][1] = 100
dp[0][2] = 100
dp[0][3] = 200

for i in range(N):
    for j in range(4):
        if S[i] == 'L':
            if j % 2 == 0:
                dp[i+1][j] = min(dp[i+1][j], dp[i][j] + abs(A[i] - A[i-1]))
            else:
                dp[i+1][j] = min(dp[i+1][j], dp[i][j] + abs(A[i] - A[i-1] + 1))
        else:
            if j % 2 == 0:
                dp[i+1][j] = min(dp[i+1][j], dp[i][j] + abs(A[i] - A[i-1] + 1))
            else:
                dp[i+1][j] = min(dp[i+1][j], dp[i][j] + abs(A[i] - A[i-1]))
        if j < 3:
            if S[i] == 'L':
                dp[i+1][j+1] = min(dp[i+1][j+1], dp[i][j] + abs(A[i] - A[i-1] + 100))
            else:
                dp[i+1][j+1] = min(dp[i+1][j+1], dp[i][j] + abs(A[i] - A[i-1] - 99))

print(min(dp[N]))