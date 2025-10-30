n, m = map(int, input().split())
s = [input() for _ in range(n)]
dp = [768] * 768
dp[0] = 0
for i in s:
    num = 0
    for j in range(m):
        if i[j] == 'o':
            num |= 1 << j
    for k in range(1024):
        dp[k | num] = min(dp[k | num], dp[k] + 1)
print(dp[767])