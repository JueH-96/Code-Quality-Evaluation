# YOUR CODE HERE
MOD = 998244353

def solve():
    N, M, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    # dp[length_mod_M][xor_value] = count of subsequences
    dp = [{} for _ in range(M)]
    dp[0][0] = 1  # empty subsequence (length 0, xor 0)
    
    result = 0
    
    for a in A:
        new_dp = [{} for _ in range(M)]
        
        # Copy current state (not including current element)
        for i in range(M):
            for xor_val, count in dp[i].items():
                new_dp[i][xor_val] = new_dp[i].get(xor_val, 0) + count
                new_dp[i][xor_val] %= MOD
        
        # Include current element
        for i in range(M):
            for xor_val, count in dp[i].items():
                new_len = (i + 1) % M
                new_xor = xor_val ^ a
                new_dp[new_len][new_xor] = new_dp[new_len].get(new_xor, 0) + count
                new_dp[new_len][new_xor] %= MOD
        
        dp = new_dp
    
    # Calculate result - only count subsequences with length multiple of M (excluding empty)
    for xor_val, count in dp[0].items():
        if xor_val == 0:
            # This includes the empty subsequence, so subtract 1
            actual_count = (count - 1) % MOD
        else:
            actual_count = count % MOD
        
        if actual_count > 0:
            contribution = (pow(xor_val, K, MOD) * actual_count) % MOD
            result = (result + contribution) % MOD
    
    print(result)

solve()