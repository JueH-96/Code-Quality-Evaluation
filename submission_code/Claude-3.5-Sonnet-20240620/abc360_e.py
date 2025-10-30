# YOUR CODE HERE
MOD = 998244353

def mod_inverse(x):
    return pow(x, MOD - 2, MOD)

def solve(N, K):
    if N == 1:
        return 1

    p = (N - 1) * mod_inverse(N * N)
    q = 1 - p
    
    p_k = pow(p, K, MOD)
    q_k = pow(q, K, MOD)
    
    numerator = (N * q_k - q_k + 1) % MOD
    denominator = mod_inverse((1 - p_k + q_k) % MOD)
    
    return (numerator * denominator) % MOD

N, K = map(int, input().split())
print(solve(N, K))