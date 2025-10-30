def solve():
    MOD = 998244353
    N = int(input())
    A = list(map(int, input().split()))
    
    # dp[i][j] represents number of ways to arrange first i numbers
    # where i-th position has value j
    dp = [[0] * (N+1) for _ in range(N+1)]
    
    # Initialize for first position
    dp[1][1] = 1
    
    # For each position
    for i in range(2, N+1):
        # For each possible value at position i
        for j in range(1, i+1):
            # Check if we can place j at position i
            valid = True
            
            # Check condition P_j > P_i for any j where A_i < j < i
            for k in range(A[i-1]+1, i):
                if k < j:
                    valid = False
                    break
            
            # Check condition P_{A_i} < P_i if A_i > 0
            if A[i-1] > 0 and j <= dp[A[i-1]][0]:
                valid = False
            
            if valid:
                # For each possible value at position i-1
                for prev in range(1, i):
                    if dp[i-1][prev] > 0:
                        # Add this possibility
                        dp[i][j] = (dp[i][j] + dp[i-1][prev]) % MOD
        
        # Store the minimum value used at position i in dp[i][0]
        min_val = float('inf')
        for j in range(1, i+1):
            if dp[i][j] > 0:
                min_val = min(min_val, j)
        dp[i][0] = min_val
    
    # Sum all possibilities for the last position
    result = 0
    for j in range(1, N+1):
        result = (result + dp[N][j]) % MOD
    
    print(result)

solve()