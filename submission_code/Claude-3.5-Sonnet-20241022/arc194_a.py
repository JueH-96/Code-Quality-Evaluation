def solve():
    # Read input
    N = int(input())
    A = list(map(int, input().split()))
    
    # dp[i][j] represents the maximum possible sum after processing first i elements
    # and having j elements in the sequence S
    dp = [[-float('inf')] * (N + 1) for _ in range(N + 1)]
    dp[0][0] = 0  # Base case: empty sequence has sum 0
    
    # For each position
    for i in range(N):
        # For each possible length of sequence before current operation
        for j in range(i + 1):
            if dp[i][j] == -float('inf'):
                continue
                
            # Option 1: Append current element
            dp[i + 1][j + 1] = max(dp[i + 1][j + 1], dp[i][j] + A[i])
            
            # Option 2: Delete last element (if sequence is not empty)
            if j > 0:
                dp[i + 1][j - 1] = max(dp[i + 1][j - 1], dp[i][j])
    
    # Find maximum value in the last row
    max_sum = max(dp[N])
    print(max_sum)

# Run the solution
solve()