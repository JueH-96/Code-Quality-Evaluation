# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    balls = []
    
    index = 1
    for i in range(N):
        x = int(data[index])
        y = int(data[index + 1])
        balls.append((x, y))
        index += 2
    
    # Sort balls by X, then by Y
    balls.sort()
    
    # dp[i] will store the number of ways to have a set ending with ball i
    dp = [0] * N
    
    # Initialize dp[i] for each ball
    for i in range(N):
        dp[i] = 1  # Each ball can be a set by itself
    
    # Calculate dp[i] using previously calculated dp[j]
    for i in range(N):
        for j in range(i):
            if balls[j][1] < balls[i][1]:
                dp[i] = (dp[i] + dp[j]) % MOD
    
    # The answer is the sum of all dp[i] values
    answer = sum(dp) % MOD
    print(answer)