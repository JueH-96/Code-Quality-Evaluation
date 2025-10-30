import math
from typing import List
from collections import defaultdict

MOD = 10**9 + 7

def compute_mobius(max_n):
    mu = [1] * (max_n + 1)
    is_prime = [True] * (max_n + 1)
    for p in range(2, max_n + 1):
        if is_prime[p]:
            for multiple in range(p, max_n + 1, p):
                is_prime[multiple] = False
                mu[multiple] *= -1
            p_square = p * p
            for multiple in range(p_square, max_n + 1, p_square):
                mu[multiple] = 0
    return mu

# Precompute divisors for numbers up to 200
divisors = [[] for _ in range(201)]
for i in range(1, 201):
    for j in range(i, 201, i):
        divisors[j].append(i)

mu = compute_mobius(200)

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        total = 0
        for g in range(1, 201):
            S_g = [x for x in nums if x % g == 0]
            if not S_g:
                continue
            X = [x // g for x in S_g]
            # Compute frequency of divisors
            freq = defaultdict(int)
            for x in X:
                for d in divisors[x]:
                    freq[d] += 1
            ans_x = 0
            for d in range(1, 201):
                if mu[d] == 0:
                    continue
                for e in range(1, 201):
                    if mu[e] == 0:
                        continue
                    lcm_de = (d * e) // math.gcd(d, e)
                    c_both = freq.get(lcm_de, 0)
                    c_d = freq.get(d, 0) - c_both
                    c_e = freq.get(e, 0) - c_both
                    # Compute term
                    pow3 = pow(3, c_both, MOD)
                    pow2_cd = pow(2, c_d + c_e, MOD)
                    term1 = (pow3 * pow2_cd) % MOD
                    pow2_ce = pow(2, c_both + c_e, MOD)
                    pow2_cd_ = pow(2, c_both + c_d, MOD)
                    term = (term1 - pow2_ce - pow2_cd_ + 1) % MOD
                    term = term * mu[d] % MOD
                    term = term * mu[e] % MOD
                    ans_x = (ans_x + term) % MOD
            total = (total + ans_x) % MOD
        return total % MOD