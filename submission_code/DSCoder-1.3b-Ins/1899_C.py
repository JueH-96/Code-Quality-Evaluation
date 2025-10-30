# YOUR CODE HERE

t = int(input().strip())
for _ in range(t):
    n = int(input().strip())
    a = list(map(int, input().strip().split()))
    dp = [0]*n
    dp[0] = a[0]
    for i in range(1,n):
        dp[i] = max(a[i], dp[i-1]+a[i])
    print(max(dp))