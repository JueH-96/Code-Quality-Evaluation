MOD = 998244353

def compute_sum_v_p(p, M):
    sum_v = 0
    for x in range(1, M + 1):
        temp = x
        while temp % p == 0:
            sum_v += 1
            temp = temp // p
    return sum_v

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    
    if M == 1:
        print(N % MOD)
        return
    
    primes = []
    for p in [2, 3, 5, 7, 11, 13]:
        if p <= M:
            primes.append(p)
    
    product = 1
    for p in primes:
        sum_v_p = compute_sum_v_p(p, M)
        
        inv_M = pow(M, MOD - 2, MOD)
        E_v = (sum_v_p * inv_M) % MOD
        
        if M == 1:
            T_mod = N % MOD
        else:
            numerator = pow(M, N + 1, MOD * (M - 1)) - M
            numerator %= MOD * (M - 1)
            denominator = (M - 1) ** 2
            inv_denominator = pow(denominator, MOD - 2, MOD)
            T_mod = (numerator * inv_denominator) % MOD
        
        if M == 1:
            S_mod = (N * (N + 1) // 2) % MOD
        else:
            if N == 0:
                S_mod = 0
            else:
                term1 = (N + 1) % MOD
                term2 = pow(M, N, MOD)
                term2 = (MOD - term2) % MOD
                term3 = pow(N, 1, MOD)
                term4 = pow(M, N + 1, MOD)
                numerator = (M * (1 - (N + 1) * term2 + N * term4)) % MOD
                denominator = (1 - M) ** 2 % MOD
                inv_denominator = pow(denominator, MOD - 2, MOD)
                S_mod = (numerator * inv_denominator) % MOD
        
        if T_mod == 0:
            E = 0
        else:
            E = (E_v * S_mod) % MOD
            inv_T_mod = pow(T_mod, MOD - 2, MOD)
            E = (E * inv_T_mod) % MOD
        
        S_p = (T_mod * (E + 1)) % MOD
        
        product = (product * S_p) % MOD
    
    print(product % MOD)

if __name__ == '__main__':
    main()