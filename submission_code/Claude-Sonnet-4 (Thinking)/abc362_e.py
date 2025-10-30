n = int(input())
a = list(map(int, input().split()))

MOD = 998244353

answers = [0] * n

# k = 1: all single elements
answers[0] = n

if n >= 2:
    # k = 2: all pairs of elements
    answers[1] = n * (n - 1) // 2 % MOD

if n >= 3:
    # dp[i][j][k] = number of arithmetic subsequences of length k ending at positions i and j (i < j)
    dp = [[[0 for _ in range(n + 1)] for _ in range(n)] for _ in range(n)]

    # Base case: any two elements form an arithmetic sequence of length 2
    for i in range(n):
        for j in range(i + 1, n):
            dp[i][j][2] = 1

    # Fill DP table
    for k in range(3, n + 1):
        for i in range(1, n):
            for j in range(i + 1, n):
                for p in range(i):
                    if 2 * a[i] == a[p] + a[j]:
                        dp[i][j][k] = (dp[i][j][k] + dp[p][i][k-1]) % MOD

    for k in range(3, n + 1):
        count = 0
        for i in range(1, n):
            for j in range(i + 1, n):
                count = (count + dp[i][j][k]) % MOD
        answers[k - 1] = count

print(' '.join(map(str, answers)))