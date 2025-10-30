import math
from typing import List

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        MAX_VAL = 200
        N = len(nums)

        # Precompute Mobius function using a linear sieve
        mu = [0] * (MAX_VAL + 1)
        lp = [0] * (MAX_VAL + 1)
        primes = []
        mu[1] = 1
        for i in range(2, MAX_VAL + 1):
            if lp[i] == 0:
                lp[i] = i
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if p > lp[i] or i * p > MAX_VAL:
                    break
                lp[i * p] = p
                if p == lp[i]:
                    mu[i * p] = 0
                else:
                    mu[i * p] = -mu[i]

        # Precompute powers of 2 and 3
        pow2 = [1] * (N + 1)
        pow3 = [1] * (N + 1)
        for i in range(1, N + 1):
            pow2[i] = (pow2[i-1] * 2) % MOD
            pow3[i] = (pow3[i-1] * 3) % MOD

        nums_counts = [0] * (MAX_VAL + 1)
        for x in nums:
            nums_counts[x] += 1
        
        total_pairs = 0
        
        # Iterate over all possible common GCDs `g`
        for g in range(1, MAX_VAL + 1):
            # Subproblem: count pairs of subsequences with GCD=1 from A = {x/g | x in nums, g|x}
            max_A = MAX_VAL // g
            if max_A == 0:
                continue

            A_counts = [0] * (max_A + 1)
            num_elements_in_A = 0
            for k in range(1, max_A + 1):
                count = nums_counts[k * g]
                if count > 0:
                    A_counts[k] = count
                    num_elements_in_A += count
            
            # Need at least two elements to form two non-empty subsequences
            if num_elements_in_A < 2:
                continue

            # count_d[d] = number of elements in multiset A that are multiples of d
            count_d = [0] * (max_A + 1)
            for d in range(1, max_A + 1):
                for k in range(d, max_A + 1, d):
                    count_d[d] += A_counts[k]
            
            pairs_for_g = 0
            # 2D inclusion-exclusion to find pairs with gcd(s1)=1 and gcd(s2)=1
            for d1 in range(1, max_A + 1):
                if mu[d1] == 0:
                    continue
                for d2 in range(1, max_A + 1):
                    if mu[d2] == 0:
                        continue
                    
                    l = (d1 * d2) // math.gcd(d1, d2)
                    
                    c_d1 = count_d[d1]
                    c_d2 = count_d[d2]
                    c_l = 0
                    if l <= max_A:
                        c_l = count_d[l]
                    
                    # P(d1,d2) is the number of pairs (s1,s2) s.t. d1|gcd(s1), d2|gcd(s2)
                    # Total ways to partition elements into s1_pot, s2_pot, neither
                    term_total = (pow2[c_d1 - c_l] * pow2[c_d2 - c_l]) % MOD
                    term_total = (term_total * pow3[c_l]) % MOD

                    # Subtract cases where s1 or s2 is empty
                    term = (term_total - pow2[c_d1] - pow2[c_d2] + 1)
                    term %= MOD
                    
                    # Apply Mobius coefficients: mu(d1)*mu(d2)*term
                    contrib = term
                    if mu[d1] == -1:
                        contrib = -contrib
                    if mu[d2] == -1:
                        contrib = -contrib
                    
                    pairs_for_g = (pairs_for_g + contrib) % MOD
            
            total_pairs = (total_pairs + pairs_for_g) % MOD
            
        return (total_pairs + MOD) % MOD