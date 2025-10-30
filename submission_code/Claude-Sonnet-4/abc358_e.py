# YOUR CODE HERE
MOD = 998244353

def solve():
    K = int(input())
    C = list(map(int, input().split()))
    
    # dp[j] = coefficient of x^j in the generating function
    dp = [0] * (K + 1)
    dp[0] = 1  # coefficient of x^0 is 1 (empty string)
    
    # For each letter
    for i in range(26):
        new_dp = [0] * (K + 1)
        
        # For each current power of x
        for j in range(K + 1):
            if dp[j] == 0:
                continue
                
            # Add contributions from using 0 to C[i] copies of letter i
            for count in range(min(C[i] + 1, K - j + 1)):
                if j + count <= K:
                    new_dp[j + count] = (new_dp[j + count] + dp[j]) % MOD
        
        dp = new_dp
    
    # Sum coefficients for lengths 1 to K
    result = 0
    for length in range(1, K + 1):
        result = (result + dp[length]) % MOD
    
    print(result)

solve()