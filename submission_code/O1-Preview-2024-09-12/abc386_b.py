# YOUR CODE HERE
S = input().strip()

BUTTONS = set(['0','1','2','3','4','5','6','7','8','9','00'])

n = len(S)

INF = float('inf')

dp = [INF] * (n+1)

dp[0]=0

for i in range(1,n+1):
    dp[i]=INF
    for l in [1,2]:
        s = i - l
        if s >=0:
            substr = S[s:i]
            if substr in BUTTONS:
                dp[i] = min(dp[i], dp[s] + 1)

print(dp[n])