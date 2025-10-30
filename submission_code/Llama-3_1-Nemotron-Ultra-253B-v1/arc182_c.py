MOD = 998244353

def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    
    # Function to generate primes up to M using sieve
    def sieve(max_n):
        if max_n < 2:
            return []
        is_prime = [True] * (max_n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(max_n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, max_n+1, i):
                    is_prime[j] = False
        primes = [i for i, val in enumerate(is_prime) if val]
        return primes
    
    primes = sieve(M)
    
    # Precompute for each prime p, the sum of v_p(x) for x in 1..M
    prime_sums = []
    for p in primes:
        s = 0
        for x in range(1, M+1):
            cnt = 0
            while x % p == 0:
                cnt += 1
                x //= p
            s += cnt
        prime_sums.append(s)
    
    # Compute sum_geo = sum_{k=1}^N M^k mod MOD
    if M == 1:
        sum_geo = N % MOD
    else:
        # sum_geo = M*(M^N - 1) / (M-1) mod MOD
        Mn = pow(M, N, MOD)
        numerator = (Mn - 1) % MOD
        inv_M_1 = pow(M-1, MOD-2, MOD)
        sum_geo = M * numerator % MOD
        sum_geo = sum_geo * inv_M_1 % MOD
    
    # Compute sum_deriv = sum_{k=1}^N k*M^{k-1} mod MOD
    if M == 1:
        sum_deriv = N % MOD
    else:
        # sum_deriv = (M^N*(N*(M-1) - 1) + 1) / (M-1)^2 mod MOD
        Mn = pow(M, N, MOD)
        term1 = Mn * (N % MOD) % MOD
        term1 = term1 * (M-1) % MOD
        term2 = Mn * 1 % MOD
        numerator = (term1 - term2 + 1) % MOD
        inv_M_1_sq = pow((M-1)**2 % MOD, MOD-2, MOD)
        sum_deriv = numerator * inv_M_1_sq % MOD
    
    # Compute the product of (sum_geo + s_p * sum_deriv) for each prime p
    result = 1
    for s_p in prime_sums:
        term = (sum_geo + s_p * sum_deriv) % MOD
        result = result * term % MOD
    
    print(result % MOD)

if __name__ == "__main__":
    main()