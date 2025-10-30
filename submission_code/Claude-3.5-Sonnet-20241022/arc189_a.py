def solve():
    MOD = 998244353
    
    # Read input
    N = int(input())
    A = list(map(int, input().split()))
    
    # Initial state: i mod 2
    initial = [i % 2 for i in range(N)]
    
    # dp[i][j] represents number of ways to achieve target state
    # for positions from i to j inclusive
    dp = [[0] * N for _ in range(N)]
    
    # Initialize dp for single positions
    for i in range(N):
        if initial[i] == A[i]:
            dp[i][i] = 1
    
    # For each length of subarray
    for length in range(2, N + 1):
        # For each starting position
        for i in range(N - length + 1):
            j = i + length - 1  # ending position
            
            # If initial state matches target state for this range
            if all(initial[k] == A[k] for k in range(i, j + 1)):
                dp[i][j] = 1
            
            # Try all possible operations
            for l in range(i, j):
                for r in range(l + 2, j + 1):
                    # Check if operation is valid in initial state
                    if initial[l] == initial[r]:
                        valid = True
                        for k in range(l + 1, r):
                            if initial[k] == initial[l]:
                                valid = False
                                break
                        
                        if valid:
                            # Create new state after operation
                            new_state = initial.copy()
                            for k in range(l + 1, r):
                                new_state[k] = new_state[l]
                            
                            # Check if we can achieve target state from new state
                            matches_target = True
                            for k in range(i, j + 1):
                                if new_state[k] != A[k]:
                                    matches_target = False
                                    break
                            
                            if matches_target:
                                # Add contribution from this operation
                                dp[i][j] = (dp[i][j] + 1) % MOD
    
    print(dp[0][N-1])

solve()