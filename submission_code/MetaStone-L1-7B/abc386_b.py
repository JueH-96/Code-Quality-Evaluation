S = input().strip()
n = len(S)
dp = [float('inf')] * (n + 1)
dp[0] = 0

for j in range(n):
    if dp[j] == float('inf'):
        continue
    # Press one button
    if j + 1 <= n:
        if dp[j + 1] > dp[j] + 1:
            dp[j + 1] = dp[j] + 1
    # Press two buttons if possible
    if j + 2 <= n:
        if S[j] == '0' and S[j + 1] == '0':
            if dp[j + 2] > dp[j] + 1:
                dp[j + 2] = dp[j] + 1

print(dp[n])