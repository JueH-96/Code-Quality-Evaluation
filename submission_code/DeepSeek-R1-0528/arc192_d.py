import math
from collections import defaultdict
import sys

MOD = 998244353
MOD1 = MOD - 1

def factorize(a):
    factors = {}
    n = a
    p = 2
    while p * p <= n:
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
        p += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:1+n-1])) if n > 1 else []
    
    factorizations = []
    for a in A:
        if a == 1:
            factorizations.append({})
        else:
            factorizations.append(factorize(a))
    
    all_primes = set()
    for fac in factorizations:
        for p in fac:
            all_primes.add(p)
    prime_occurrences = defaultdict(list)
    for idx, fac in enumerate(factorizations):
        for p, exp in fac.items():
            prime_occurrences[p].append((idx, exp))
            
    result = 1
    for p in all_primes:
        occurrences = prime_occurrences[p]
        k = len(occurrences)
        total_p = 0
        for assignment_index in range(1 << k):
            M_p = 0
            e = [0] * n
            for idx_in_occurrence in range(k):
                j_index, v_j = occurrences[idx_in_occurrence]
                x_bit = 1 if (assignment_index >> idx_in_occurrence) & 1 else 0
                M_p += (n - 1 - j_index) * v_j
                if x_bit == 1:
                    linear = (2 * j_index + 2 - n) * v_j
                    M_p += linear
                for i in range(n):
                    if i <= j_index:
                        if x_bit == 1:
                            e[i] += v_j
                    else:
                        if x_bit == 0:
                            e[i] += v_j
            min_exponent = min(e) if n > 0 else 0
            exp_score = M_p - n * min_exponent
            exp_reduced = exp_score % MOD1
            if exp_reduced < 0:
                exp_reduced += MOD1
            value = pow(p, exp_reduced, MOD)
            total_p = (total_p + value) % MOD
        result = (result * total_p) % MOD
    print(result)

if __name__ == "__main__":
    main()