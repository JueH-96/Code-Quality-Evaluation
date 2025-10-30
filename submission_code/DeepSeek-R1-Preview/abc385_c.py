n = int(input())
H = list(map(int, input().split()))
max_count = 1  # At least one building can be chosen
dp = [[1] * (n) for _ in range(n)]

for i in range(n-2, -1, -1):
    for d in range(1, n - i):
        j = i + d
        if j >= n:
            dp[i][d] = 1
        else:
            if H[i] == H[j]:
                dp[i][d] = 1 + dp[j][d]
            else:
                dp[i][d] = 1
        if dp[i][d] > max_count:
            max_count = dp[i][d]

print(max_count)