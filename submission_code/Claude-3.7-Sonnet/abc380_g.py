def solve(N, K, P):
    MOD = 998244353
    
    numerator = 0
    denominator = 2 * (N-K+1)
    
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            # Number of starting positions where both i and j are in the shuffled region
            M = min(i, N-K+1) - max(1, j-K+1) + 1
            if min(i, N-K+1) < max(1, j-K+1):
                M = 0
            
            # Numerator for the probability of an inversion
            is_inverted = 1 if P[i-1] > P[j-1] else 0
            p_inv_numerator = M + 2 * (N-K+1 - M) * is_inverted
            numerator += p_inv_numerator
    
    # Find modular inverse
    inv_denominator = pow(denominator, MOD-2, MOD)
    result = (numerator % MOD * inv_denominator) % MOD
    
    return result

# Reading input
N, K = map(int, input().split())
P = list(map(int, input().split()))

# Compute and print the result
print(solve(N, K, P))