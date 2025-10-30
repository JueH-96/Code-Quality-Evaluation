def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:]))
    
    MOD = 998244353
    
    # Maximum possible XOR value given the constraints
    MAX_XOR = 1 << 20
    
    # dp[x][l] = number of subsequences with XOR x and length l
    dp = [[0] * (N + 1) for _ in range(MAX_XOR)]
    
    # Initial state: empty subsequence with XOR 0 and length 0
    dp[0][0] = 1
    
    # Process each element in A
    for a in A:
        # We need to update dp table, so we work with a copy to avoid overwriting
        new_dp = [row[:] for row in dp]
        
        for x in range(MAX_XOR):
            for l in range(N):
                if dp[x][l] > 0:
                    # Add the current element a to subsequences with XOR x and length l
                    new_x = x ^ a
                    new_l = l + 1
                    new_dp[new_x][new_l] = (new_dp[new_x][new_l] + dp[x][l]) % MOD
        
        dp = new_dp
    
    # Calculate the result
    result = 0
    for x in range(MAX_XOR):
        for l in range(M, N + 1, M):
            if dp[x][l] > 0:
                # Calculate the score for this XOR and length
                score = pow(x, K, MOD)
                result = (result + score * dp[x][l]) % MOD
    
    print(result)