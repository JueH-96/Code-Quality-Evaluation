import math
from typing import List

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 1000000007
        n = len(nums)
        max_n = n
        
        # Compute factorial
        fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            fact[i] = (fact[i - 1] * i) % MOD
        
        # Compute inverse factorial
        inv_fact_max = pow(fact[max_n], MOD - 2, MOD)
        inv_fact = [0] * (max_n + 1)
        inv_fact[max_n] = inv_fact_max
        for i in range(max_n, 0, -1):
            inv_fact[i - 1] = (inv_fact[i] * i) % MOD
        
        # Sort nums and find unique values and frequencies
        sorted_nums = sorted(nums)
        value_freq = []
        current_val = sorted_nums[0]
        count = 1
        for num in sorted_nums[1:]:
            if num == current_val:
                count += 1
            else:
                value_freq.append((current_val, count))
                current_val = num
                count = 1
        value_freq.append((current_val, count))  # Add the last group
        m = len(value_freq)
        
        # Compute freq_sum_ge (number of elements >= value at index i)
        freq_sum_ge = [0] * m
        if m > 0:
            freq_sum_ge[m - 1] = value_freq[m - 1][1]
            for i in range(m - 2, -1, -1):
                freq_sum_ge[i] = value_freq[i][1] + freq_sum_ge[i + 1]
        
        # Compute prefix_sum_le (number of elements <= value at index i)
        prefix_sum_le = [0] * m
        if m > 0:
            prefix_sum_le[0] = value_freq[0][1]
            for i in range(1, m):
                prefix_sum_le[i] = prefix_sum_le[i - 1] + value_freq[i][1]
        
        # Define helper function for binomial sum sum_{s=1 to k} C(n, s)
        def binom_sum(n_sum, k_sum):
            if n_sum < 0 or k_sum < 0:
                return 0
            total = 0
            max_s = min(k_sum, n_sum)
            for s in range(1, max_s + 1):
                comb = (fact[n_sum] * inv_fact[s] % MOD * inv_fact[n_sum - s] % MOD) % MOD
                total = (total + comb) % MOD
            return total
        
        # Compute sum of min over all subsets with size 1 to k
        sum_min_part = 0
        for idx in range(m):
            val_idx = value_freq[idx][0]
            F_curr = freq_sum_ge[idx]
            F_next = freq_sum_ge[idx + 1] if idx + 1 < m else 0
            diff_min = (binom_sum(F_curr, k) - binom_sum(F_next, k) + MOD) % MOD
            contrib_min = (val_idx * diff_min) % MOD
            sum_min_part = (sum_min_part + contrib_min) % MOD
        
        # Compute sum of max over all subsets with size 1 to k
        sum_max_part = 0
        for idx in range(m):
            val_idx = value_freq[idx][0]
            G_curr = prefix_sum_le[idx]
            G_prev = prefix_sum_le[idx - 1] if idx > 0 else 0
            diff_max = (binom_sum(G_curr, k) - binom_sum(G_prev, k) + MOD) % MOD
            contrib_max = (val_idx * diff_max) % MOD
            sum_max_part = (sum_max_part + contrib_max) % MOD
        
        # Total sum is sum of min and max parts
        total_sum = (sum_min_part + sum_max_part) % MOD
        return total_sum