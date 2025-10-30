# YOUR CODE HERE
N, K = map(int, input().split())
if K > 0:
    A = list(map(int, input().split()))
else:
    A = []

if K == 0:
    print(0)
else:
    dp = [float('inf')] * (K + 1)
    dp[0] = 0
    
    for i in range(1, K + 1):
        if i == 1:
            dp[i] = 0  # Leave the first sock unpaired
        else:
            # Pair the (i-1)-th and i-th socks
            dp[i] = dp[i-2] + A[i-1] - A[i-2]
            # If i is odd, we can also leave the i-th sock unpaired
            if i % 2 == 1:
                dp[i] = min(dp[i], dp[i-1])
    
    print(dp[K])