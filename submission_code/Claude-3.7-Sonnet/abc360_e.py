def expected_position(N, K):
    MOD = 998244353
    
    # If K=0, the black ball is at position 1
    if K == 0:
        return 1
    
    # Calculate (1 - 2/N)^K
    # First, we compute (N-2)^K and N^K
    alpha_numerator = pow(N - 2, K, MOD)
    alpha_denominator = pow(N, K, MOD)
    
    # Calculate the modular inverse of alpha_denominator
    inv_alpha_denominator = pow(alpha_denominator, MOD - 2, MOD)
    
    # Calculate (N-2)^K / N^K
    alpha_K = (alpha_numerator * inv_alpha_denominator) % MOD
    
    # Calculate (N+1)/2 + (1-N)/2 * alpha^K
    first_term = ((N + 1) * pow(2, MOD - 2, MOD)) % MOD
    
    # Handle (1-N) in modular arithmetic
    one_minus_N = (1 - N) % MOD
    
    second_term = (one_minus_N * alpha_K * pow(2, MOD - 2, MOD)) % MOD
    
    # Final result
    result = (first_term + second_term) % MOD
    
    return result

# Read input and print output
N, K = map(int, input().split())
print(expected_position(N, K))