import math
from typing import List

class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 1000000007
        n = len(nums)
        
        # Sort the nums array
        sorted_nums = sorted(nums)
        
        # Find unique values and their frequencies
        vals = []
        freqs = []
        if n > 0:
            current_val = sorted_nums[0]
            count = 1
            for num in sorted_nums[1:]:
                if num == current_val:
                    count += 1
                else:
                    vals.append(current_val)
                    freqs.append(count)
                    current_val = num
                    count = 1
            vals.append(current_val)  # Add the last group
            freqs.append(count)
        
        k = len(vals)
        
        # Compute cumulative sum of frequencies
        cum_sum_list = [0]
        accum = 0
        for freq in freqs:
            accum += freq
            cum_sum_list.append(accum)
        
        # Compute modular inverse of 2
        inv2 = pow(2, MOD - 2, MOD)
        
        # Precompute powers of 2 and inverse powers of 2 up to n
        pow2_list = [0] * (n + 1)
        pow2_list[0] = 1
        for exp in range(1, n + 1):
            pow2_list[exp] = (pow2_list[exp - 1] * 2 % MOD) % MOD
        
        inv_pow2_list = [0] * (n + 1)
        inv_pow2_list[0] = 1
        for exp in range(1, n + 1):
            inv_pow2_list[exp] = (inv_pow2_list[exp - 1] * inv2 % MOD) % MOD
        
        # Compute prefix sum of E_j
        prefix_sum_E = [0]  # Start with sum 0
        cum_E = 0
        for j in range(k):
            # Compute diff_inv
            diff_inv = (inv_pow2_list[cum_sum_list[j]] - inv_pow2_list[cum_sum_list[j + 1]] + MOD) % MOD
            # Compute E_j
            E_j_val = ((vals[j] % MOD) * (diff_inv % MOD) % MOD) % MOD
            # Add to cumulative sum
            cum_E = (cum_E + E_j_val) % MOD
            # Append the cumulative sum up to this j
            prefix_sum_E.append(cum_E)
        
        # Compute sum_min for each i
        sum_min_list = []
        for i in range(k):
            sum_min_i = (pow2_list[cum_sum_list[i + 1]] * (prefix_sum_E[i + 1] % MOD) % MOD) % MOD
            sum_min_list.append(sum_min_i)
        
        # Compute the total sum
        total_sum = 0
        for m in range(k):
            pow_val = (vals[m] * vals[m] % MOD) % MOD  # Squared value modulo MOD
            current_sum_min = sum_min_list[m]
            if m == 0:
                prev_sum_min = 0
            else:
                prev_sum_min = sum_min_list[m - 1]
            diff_sum_min = (current_sum_min - prev_sum_min + MOD) % MOD
            total_sum = (total_sum + (pow_val * diff_sum_min % MOD) % MOD) % MOD
        
        return total_sum