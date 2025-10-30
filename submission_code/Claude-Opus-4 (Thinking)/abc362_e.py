n = int(input())
a = list(map(int, input().split()))

MOD = 998244353

# dp[len][last][second_last] = number of arithmetic subsequences
dp = [[[0] * n for _ in range(n)] for _ in range(n + 1)]

# Length 2: all pairs
for i in range(n):
    for j in range(i + 1, n):
        dp[2][j][i] = 1

# Length >= 3
for length in range(3, n + 1):
    for j in range(n):
        for i in range(j):
            for k in range(i):
                if a[i] - a[k] == a[j] - a[i]:
                    dp[length][j][i] = (dp[length][j][i] + dp[length - 1][i][k]) % MOD

# Collect answers
answers = []
for length in range(1, n + 1):
    total = 0
    if length == 1:
        total = n
    else:
        for j in range(n):
            for i in range(j):
                total = (total + dp[length][j][i]) % MOD
    answers.append(total)

print(' '.join(map(str, answers)))