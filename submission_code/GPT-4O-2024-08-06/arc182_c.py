def number_of_divisors(x):
    divisors = 0
    for i in range(1, int(x**0.5) + 1):
        if x % i == 0:
            divisors += 1
            if i != x // i:
                divisors += 1
    return divisors

def solve():
    import sys
    input = sys.stdin.read
    data = input().strip()
    N, M = map(int, data.split())
    
    MOD = 998244353
    
    # Precompute the number of divisors for each number from 1 to M
    divisors_count = [0] * (M + 1)
    for i in range(1, M + 1):
        divisors_count[i] = number_of_divisors(i)
    
    # Calculate the sum of scores for sequences of length 1 to N
    total_score = 0
    power_of_M = 1
    
    for k in range(1, M + 1):
        power_of_M = (power_of_M * M) % MOD
        total_score = (total_score + power_of_M * divisors_count[k]) % MOD
    
    # Calculate the geometric series sum for sequences of length 1 to N
    if N == 1:
        print(total_score)
        return
    
    # Geometric series sum: (M^1 + M^2 + ... + M^N) = M * (M^N - 1) / (M - 1)
    # Using modular arithmetic to calculate this
    def mod_exp(base, exp, mod):
        result = 1
        while exp > 0:
            if exp % 2 == 1:
                result = (result * base) % mod
            base = (base * base) % mod
            exp //= 2
        return result
    
    M_power_N = mod_exp(M, N, MOD)
    geometric_sum = (M * (M_power_N - 1) * pow(M - 1, MOD - 2, MOD)) % MOD
    
    # Multiply the total score by the geometric sum
    final_result = (total_score * geometric_sum) % MOD
    
    print(final_result)