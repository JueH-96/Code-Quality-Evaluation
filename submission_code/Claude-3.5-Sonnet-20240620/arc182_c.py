# YOUR CODE HERE
MOD = 998244353

def pow_mod(base, exp, mod):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

def solve(N, M):
    # Precompute prime factors of numbers from 1 to M
    prime_factors = [[] for _ in range(M + 1)]
    for i in range(2, M + 1):
        if not prime_factors[i]:
            for j in range(i, M + 1, i):
                factor = i
                while j % factor == 0:
                    prime_factors[j].append(i)
                    factor *= i

    # Calculate the contribution of each prime to the final result
    prime_contrib = [0] * (M + 1)
    for i in range(1, M + 1):
        factor_count = [0] * (M + 1)
        for factor in prime_factors[i]:
            factor_count[factor] += 1
        
        contrib = 1
        for count in factor_count:
            if count > 0:
                contrib = (contrib * (count + 1)) % MOD
        prime_contrib[i] = contrib

    # Calculate the sum of all contributions
    total_contrib = sum(prime_contrib) % MOD

    # Calculate the final result using geometric series
    if N == 1:
        return total_contrib
    else:
        numerator = (pow_mod(M, N + 1, MOD) - M + MOD - 1) % MOD
        denominator = pow_mod(M - 1, MOD - 2, MOD)
        return (total_contrib * numerator * denominator) % MOD

N, M = map(int, input().split())
print(solve(N, M))