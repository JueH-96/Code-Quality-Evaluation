n = int(input().strip())
H = list(map(int, input().split()))

if n == 1:
    print(1)
    exit(0)

dp = [[0] * j for j in range(n)]
ans = 1

for j in range(1, n):
    for i in range(j):
        if H[i] == H[j]:
            k = 2 * i - j
            if k >= 0 and k < i and H[k] == H[i]:
                dp[j][i] = dp[i][k] + 1
            else:
                dp[j][i] = 2
            if dp[j][i] > ans:
                ans = dp[j][i]

print(ans)