MOD = 998244353

def solve():
    N, M = map(int, input().split())

    if N == 1:
        # Covers N=1, M=1 and N=1, M>1
        # Expected ops = M^2.
        ans = pow(M, 2, MOD)
        print(ans)
        return

    # From here, N > 1 is guaranteed.
    
    # Precompute H_N = sum_{i=1 to N} (1/i) mod MOD
    # inv[i] stores i^(-1) mod MOD
    # H_arr[i] stores H_i mod MOD (H_arr to avoid conflict with H_N variable name)
    
    inv = [0] * (N + 1) # inv[0] unused
    H_arr = [0] * (N + 1)   # H_arr[0] unused
    
    inv[1] = 1
    H_arr[1] = 1 
    
    for i in range(2, N + 1):
        inv[i] = MOD - (MOD // i) * inv[MOD % i] % MOD
        H_arr[i] = (H_arr[i-1] + inv[i]) % MOD
    
    H_N = H_arr[N] # This is H_N for the N read from input

    if M == 1:
        # Case N > 1 and M = 1
        # Expected ops = 2*N*H_N.
        term_2N = (2 * N) % MOD
        ans = (term_2N * H_N) % MOD
        print(ans)
        return
        
    # Case N > 1 and M > 1
    # Formula: M^2 * (2*N*H_N - (N-1))
    
    term_2NHN = (2 * N % MOD * H_N) % MOD
    
    # factor = (2*N*H_N - (N-1))
    # To compute (A - B) % MOD, use (A - B % MOD + MOD) % MOD
    term_N_minus_1 = (N - 1) # N > 1, so N-1 >= 1
    factor = (term_2NHN - term_N_minus_1 % MOD + MOD) % MOD
    
    M_squared = pow(M, 2, MOD)
    ans = (M_squared * factor) % MOD
    
    print(ans)

solve()