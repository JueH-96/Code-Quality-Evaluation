# YOUR CODE HERE
S = input().strip()
n = len(S)
dp = [float('inf')] * (n + 1)
dp[0] = 0

for i in range(n):
    if dp[i] == float('inf'):
        continue
    # Press a single digit
    if i + 1 <= n:
        dp[i+1] = min(dp[i+1], dp[i] + 1)
    # Press '00' if the next two characters are '00'
    if i + 2 <= n and S[i] == '0' and S[i+1] == '0':
        dp[i+2] = min(dp[i+2], dp[i] + 1)

print(dp[n])