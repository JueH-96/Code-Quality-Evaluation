# YOUR CODE HERE
def min_time_to_reach_stage_n():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = [0] * (N)
    B = [0] * (N)
    X = [0] * (N)
    
    index = 1
    for i in range(1, N):
        A[i] = int(data[index])
        B[i] = int(data[index + 1])
        X[i] = int(data[index + 2])
        index += 3
    
    # Initialize dp array with infinity
    dp = [float('inf')] * (N + 1)
    dp[1] = 0  # Starting at stage 1 requires 0 time
    
    # Fill the dp array
    for i in range(1, N):
        # Option 1: Move to stage i+1
        if i + 1 <= N:
            dp[i + 1] = min(dp[i + 1], dp[i] + A[i])
        
        # Option 2: Jump to stage X[i]
        if X[i] <= N:
            dp[X[i]] = min(dp[X[i]], dp[i] + B[i])
    
    # The answer is the minimum time to reach stage N
    print(dp[N])