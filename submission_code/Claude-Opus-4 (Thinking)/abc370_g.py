# YOUR CODE HERE
MOD = 998244353

def get_prime_factors(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def is_good(n):
    # A number is good if it has at least one prime p ≡ 2 (mod 3) with odd exponent
    factors = get_prime_factors(n)
    for p, exp in factors.items():
        if p % 3 == 2 and exp % 2 == 1:
            return True
    return False

def solve(N, M):
    # Precompute factorials and inverse factorials
    max_val = min(N + M, 200000)
    fact = [1] * (max_val + 1)
    for i in range(1, max_val + 1):
        fact[i] = fact[i-1] * i % MOD
    
    inv_fact = [1] * (max_val + 1)
    inv_fact[max_val] = pow(fact[max_val], MOD - 2, MOD)
    for i in range(max_val - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
    
    def comb(n, r):
        if r < 0 or r > n or n >= len(fact):
            # For large n, use direct computation
            if r == 0:
                return 1
            result = 1
            for i in range(r):
                result = result * (n - i) % MOD
                result = result * pow(i + 1, MOD - 2, MOD) % MOD
            return result
        return fact[n] * inv_fact[r] % MOD * inv_fact[n - r] % MOD
    
    # For large N, we need to be smarter
    # Key insight: most good numbers up to N have small prime factors
    result = 0
    
    # Generate good numbers efficiently
    def generate_good_numbers(limit, current, min_prime, has_odd_p2):
        nonlocal result
        
        if current > limit:
            return
        
        if has_odd_p2 and current > 0:
            # Count sequences with product = current
            factors = get_prime_factors(current)
            count = 1
            for exp in factors.values():
                count = count * comb(exp + M - 1, M - 1) % MOD
            result = (result + count) % MOD
        
        # Try multiplying by primes
        primes = []
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            if p >= min_prime and current * p <= limit:
                primes.append(p)
        
        for p in primes:
            if current * p > limit:
                break
            # Add one factor of p
            new_has_odd = has_odd_p2
            if p % 3 == 2:
                # Check if adding this p makes an odd exponent
                temp = current
                exp = 0
                while temp % p == 0:
                    exp += 1
                    temp //= p
                if exp % 2 == 0:  # Currently even, will become odd
                    new_has_odd = True
            generate_good_numbers(limit, current * p, p, new_has_odd)
    
    # Start generation
    generate_good_numbers(N, 1, 2, False)
    
    return result

N, M = map(int, input().split())
print(solve(N, M))