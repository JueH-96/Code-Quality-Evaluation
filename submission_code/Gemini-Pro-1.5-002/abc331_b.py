# YOUR CODE HERE
def solve():
    n, s, m, l = map(int, input().split())
    
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    for i in range(1, n + 1):
        if i >= 6:
            dp[i] = min(dp[i], dp[i - 6] + s)
        if i >= 8:
            dp[i] = min(dp[i], dp[i - 8] + m)
        if i >= 12:
            dp[i] = min(dp[i], dp[i - 12] + l)
        if i > 0:
            dp[i] = min(dp[i], dp[i-1] + min(s//6 + (1 if s%6 > 0 else 0), m//8 + (1 if m%8 > 0 else 0), l//12 + (1 if l%12 > 0 else 0)))

    print(dp[n])

solve()