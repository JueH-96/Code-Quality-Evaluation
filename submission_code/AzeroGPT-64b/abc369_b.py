N = int(input())

hands = []
for _ in range(N):
    A, S = input().split()
    hands.append((int(A), S))

INF = 10**9
dp = [[INF] * 4 for _ in range(N + 1)]
dp[0][0] = dp[0][1] = dp[0][2] = dp[0][3] = 0

for i in range(N):
    A, S = hands[i]

    if S == 'L':
        # previous right -> left
        dp[i + 1][0] = min(dp[i + 1][0], dp[i][1] + abs(A - hands[i-1][0] if i > 0 else 100))
        dp[i + 1][2] = min(dp[i + 1][2], dp[i][3] + abs(A - hands[i-1][0] if i > 0 else 100))
        # previous left -> right
        dp[i + 1][1] = min(dp[i + 1][1], dp[i][1] + abs(A - hands[i-1][0] if i > 0 else 100))

    if S == 'R':
        # previous right -> left
        dp[i + 1][0] = min(dp[i + 1][0], dp[i][0] + abs(A - hands[i-1][0] if i > 0 else 100))
        # previous left -> left
        dp[i + 1][1] = min(dp[i + 1][1], dp[i][0] + abs(A - hands[i-1][0] if i > 0 else 100))
        # previous right -> left
        dp[i + 1][2] = min(dp[i + 1][2], dp[i][2] + abs(A - hands[i-1][0] if i > 0 else 100))

    # two hands right
    dp[i + 1][3] = min(dp[i + 1][3], dp[i][3] + abs(A - hands[i-1][0] if i > 0 else 100))
    if i > 0 and hands[i-1][1] == 'L':
        dp[i + 1][3] = min(dp[i + 1][3], dp[i][0] + abs(A - hands[i-1][0]))
        dp[i + 1][3] = min(dp[i + 1][3], dp[i][1] + abs(A - hands[i-1][0]))
        dp[i + 1][3] = min(dp[i + 1][3], dp[i][2] + abs(A - hands[i-1][0]))

print(min(dp[-1]))