import math

s = input()
n = len(s)
dp = [float('inf')] * (n + 1)
dp[0] = 0

for i in range(1, n + 1):
    if i >= 2 and s[i-2:i] == '00':
        dp[i] = min(dp[i], dp[i-2] + 1)
    if i >= 1 and s[i-1] == '0':
        dp[i] = min(dp[i], dp[i-1] + 1)
    for digit in range(1, 10):
        if i >= 1 and s[i-1] == str(digit):
            dp[i] = min(dp[i], dp[i-1] + 1)

print(dp[n])