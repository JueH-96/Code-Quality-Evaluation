MOD = 998244353

def prime_factors(n):
    i = 2
    factors = {}
    while i * i <= n:
        while (n % i) == 0:
            if i in factors:
                factors[i] += 1
            else:
                factors[i] = 1
            n //= i
        i += 1
    if n > 1:
        factors[n] = 1
    return factors

def count_divisors(factors):
    count = 1
    for exp in factors.values():
        count *= (exp + 1)
    return count

def mod_exp(base, exp, mod):
    result = 1
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp //= 2
    return result

def solve():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    N = int(data[0])
    M = int(data[1])
    
    # Precompute the number of divisors for each number from 1 to M
    divisors_count = [0] * (M + 1)
    for i in range(1, M + 1):
        factors = prime_factors(i)
        divisors_count[i] = count_divisors(factors)
    
    # Sum the scores of all sequences
    total_score = 0
    for length in range(1, N + 1):
        if length > 18:  # Avoid large powers if N is very large
            break
        # Calculate M^length % MOD
        power = mod_exp(M, length, MOD)
        # Sum the scores of sequences of this length
        score_sum = sum(divisors_count) % MOD
        total_score = (total_score + score_sum * power) % MOD
    
    # If N is very large, we need to consider the repeating pattern
    if N > 18:
        # Calculate M^19 % MOD
        power_19 = mod_exp(M, 19, MOD)
        # Sum the scores of sequences of length 19 and more
        score_sum = sum(divisors_count) % MOD
        # Geometric series sum for lengths from 19 to N
        if power_19 == 1:
            total_score = (total_score + score_sum * (N - 18)) % MOD
        else:
            # Sum of geometric series a * r^0 + a * r^1 + ... + a * r^(n-1) is a * (1 - r^n) / (1 - r)
            remaining_terms = N - 18
            geometric_sum = (score_sum * (1 - mod_exp(power_19, remaining_terms, MOD)) * mod_exp(1 - power_19, MOD - 2, MOD)) % MOD
            total_score = (total_score + geometric_sum) % MOD
    
    print(total_score)