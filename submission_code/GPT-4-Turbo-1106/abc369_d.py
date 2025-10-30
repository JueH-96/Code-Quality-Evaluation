def max_experience(N, monsters):
    # Initialize the dynamic programming table
    # dp[i][j] will store the maximum experience points Takahashi can gain after encountering the i-th monster,
    # with j indicating whether he defeated an even number of monsters (0 for even, 1 for odd).
    dp = [[0, 0] for _ in range(N+1)]
    
    for i in range(1, N+1):
        # If Takahashi lets the i-th monster go, the experience points remain the same.
        dp[i][0] = max(dp[i][0], dp[i-1][0])
        dp[i][1] = max(dp[i][1], dp[i-1][1])
        
        # If Takahashi defeats the i-th monster, he gains experience points.
        # If the number of defeated monsters before was odd, now it will be even, so he gains additional points.
        dp[i][1] = max(dp[i][1], dp[i-1][0] + monsters[i-1])
        dp[i][0] = max(dp[i][0], dp[i-1][1] + monsters[i-1] * 2)
    
    # The maximum experience points will be the maximum of the two possibilities after encountering all monsters.
    return max(dp[N])

# Read input from stdin
N = int(input().strip())
monsters = list(map(int, input().strip().split()))

# Calculate and print the answer
print(max_experience(N, monsters))