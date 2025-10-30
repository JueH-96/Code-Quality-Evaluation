n = int(input())
s = input()



dp = [0]*(n+1)

dp[0] = 0

for i, v in enumerate(s):
  if v == '0':
    dp[i+1] = dp[i] + (n-i-1)
  else:
    dp[i+1] = dp[i] + (n-i-1*-1) + 1

print(dp[-1])