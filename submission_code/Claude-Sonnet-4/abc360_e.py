MOD = 998244353

def modinv(a, m=MOD):
    return pow(a, m-2, m)

def solve(N, K):
    if N == 1:
        return 1
    
    # Transition matrix coefficients
    # p(k+1) = p(k) * (N^2 - 2N + 2)/N^2 + q(k) * 2(N-1)/N^2
    # q(k+1) = p(k) * 2/N^2 + q(k) * (N^2 - 2N + 2)/N^2 + q(k) * 2(N-2)/N^2
    #        = p(k) * 2/N^2 + q(k) * (N^2 - 2N + 2 + 2N - 4)/N^2
    #        = p(k) * 2/N^2 + q(k) * (N^2 - 2)/N^2
    
    N2 = (N * N) % MOD
    inv_N2 = modinv(N2)
    
    # Coefficients
    a11 = ((N2 - 2*N + 2) * inv_N2) % MOD
    a12 = (2 * (N-1) * inv_N2) % MOD
    a21 = (2 * inv_N2) % MOD
    a22 = ((N2 - 2) * inv_N2) % MOD
    
    # Initial state: p=1, q=0
    p, q = 1, 0
    
    # Apply K operations
    for _ in range(K):
        new_p = (p * a11 + q * a12) % MOD
        new_q = (p * a21 + q * a22) % MOD
        p, q = new_p, new_q
    
    # Expected value = p + q * (N(N+1)/2 - 1)
    sum_2_to_N = (N * (N + 1) // 2 - 1) % MOD
    expected = (p + q * sum_2_to_N) % MOD
    
    return expected

N, K = map(int, input().split())
print(solve(N, K))