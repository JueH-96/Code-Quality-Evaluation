# YOUR CODE HERE
MOD = 998244353

def solve():
    N, M, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    # dp[j][x] = number of subsequences with length ≡ j (mod M) and XOR = x
    # We'll use dictionaries to handle sparse XOR values
    dp = [dict() for _ in range(M)]
    
    # Process each element
    for a in A:
        # Create new dp state
        new_dp = [dict() for _ in range(M)]
        
        # Copy existing subsequences (not including current element)
        for j in range(M):
            for x, count in dp[j].items():
                if x not in new_dp[j]:
                    new_dp[j][x] = 0
                new_dp[j][x] = (new_dp[j][x] + count) % MOD
        
        # Add subsequences that include current element
        # Single element subsequence
        if a not in new_dp[1 % M]:
            new_dp[1 % M][a] = 0
        new_dp[1 % M][a] = (new_dp[1 % M][a] + 1) % MOD
        
        # Extend existing subsequences
        for j in range(M):
            for x, count in dp[j].items():
                new_x = x ^ a
                new_j = (j + 1) % M
                if new_x not in new_dp[new_j]:
                    new_dp[new_j][new_x] = 0
                new_dp[new_j][new_x] = (new_dp[new_j][new_x] + count) % MOD
        
        dp = new_dp
    
    # Calculate the answer
    ans = 0
    
    # Only subsequences with length ≡ 0 (mod M) contribute to the score
    for x, count in dp[0].items():
        # Score = x^K
        score = pow(x, K, MOD)
        ans = (ans + score * count) % MOD
    
    print(ans)

solve()