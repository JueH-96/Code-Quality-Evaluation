def count_divisors(n):
    count = 0
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
    return count

def mod_exp(base, exp, mod):
    result = 1
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp //= 2
    return result

def sum_of_scores(N, M):
    MOD = 998244353
    total_score = 0
    
    # Precompute the number of divisors for each number from 1 to M
    divisor_count = [count_divisors(i) for i in range(M + 1)]
    
    # Calculate the total score for sequences of length 1 to N
    for length in range(1, N + 1):
        # Each element can be from 1 to M
        # The contribution of sequences of this length is:
        # (sum of divisors from 1 to M) * M^(length - 1)
        sum_divisors = sum(divisor_count[1:M + 1]) % MOD
        contribution = sum_divisors * mod_exp(M, length - 1, MOD) % MOD
        total_score = (total_score + contribution) % MOD
    
    return total_score

import sys
input = sys.stdin.read
N, M = map(int, input().strip().split())
print(sum_of_scores(N, M))