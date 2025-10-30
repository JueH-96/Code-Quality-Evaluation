import math
from typing import List

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 1000000007
        MAX_NUM = 200
        
        # Define a helper function to compute Möbius function for a number
        def get_mu(n):
            nn = n
            count = 0
            if nn % 2 == 0:
                nn //= 2
                count += 1
                if nn % 2 == 0:
                    return 0
            j = 3
            while j * j <= nn:
                if nn % j == 0:
                    nn //= j
                    count += 1
                    if nn % j == 0:
                        return 0
                j += 2
            if nn > 1:
                count += 1
            return 1 if count % 2 == 0 else -1
        
        # Compute Möbius function array
        mu_array = [0] * (MAX_NUM + 1)
        for i in range(1, MAX_NUM + 1):
            mu_array[i] = get_mu(i)
        
        # Compute count array: cnt[k] is the number of elements in nums divisible by k
        cnt = [0] * 201  # Index 0 unused, cnt[k] for k=1 to 200
        for k in range(1, 201):
            cnt[k] = sum(1 for num in nums if num % k == 0)
        
        # Initialize total sum
        total_sum = 0
        
        # Iterate over all possible g
        for g in range(1, 201):
            sum_g = 0
            # Iterate over all d and e
            for d in range(1, 201):
                for e in range(1, 201):
                    # Compute lcm(d, e)
                    gcd_de = math.gcd(d, e)
                    lcm_de = (d * e) // gcd_de
                    # Compute k_lcm, k_d, k_e
                    k_lcm = g * lcm_de
                    k_d = g * d
                    k_e = g * e
                    # Get cnt values, cap at 200
                    cnt_lcm = 0 if k_lcm > 200 else cnt[k_lcm]
                    cnt_d_val = 0 if k_d > 200 else cnt[k_d]
                    cnt_e_val = 0 if k_e > 200 else cnt[k_e]
                    # Compute exponent for 2
                    exp2 = cnt_d_val + cnt_e_val - 2 * cnt_lcm
                    # Compute exponent modulo phi(MOD) for pow function
                    phi_mod = MOD - 1
                    exp_mod = (exp2 % phi_mod + phi_mod) % phi_mod
                    # Compute powers
                    pow2_val = pow(2, exp_mod, MOD)
                    pow3_val = pow(3, cnt_lcm, MOD)
                    pow_prod = (pow3_val * pow2_val) % MOD
                    # Compute mu product
                    mu_prod = mu_array[d] * mu_array[e]
                    # Compute term
                    term = (mu_prod * pow_prod) % MOD
                    term = (term + MOD) % MOD  # Ensure non-negative
                    # Add to sum_g
                    sum_g = (sum_g + term) % MOD
            # Add sum_g to total_sum
            total_sum = (total_sum + sum_g) % MOD
        
        return total_sum