MOD = 998244353

def mod_inverse(a, m):
    return pow(a, m - 2, m)

def solve(N, K):
    if N == 1:
        return 1
    
    # Expected position after K operations
    # E[X_K] = (N+1)/2 - (N-1)/2 * ((N-2)/N)^K
    
    # Calculate ((N-2)/N)^K mod MOD
    numerator = pow(N - 2, K, MOD)
    denominator = pow(N, K, MOD)
    denominator_inv = mod_inverse(denominator, MOD)
    
    # ((N-2)/N)^K mod MOD
    fraction = (numerator * denominator_inv) % MOD
    
    # (N-1)/2 mod MOD
    n_minus_1_half = ((N - 1) * mod_inverse(2, MOD)) % MOD
    
    # (N+1)/2 mod MOD
    n_plus_1_half = ((N + 1) * mod_inverse(2, MOD)) % MOD
    
    # E[X_K] = (N+1)/2 - (N-1)/2 * ((N-2)/N)^K
    result = (n_plus_1_half - (n_minus_1_half * fraction) % MOD + MOD) % MOD
    
    return result

# Read input
N, K = map(int, input().split())

# Solve and print result
print(solve(N, K))