from collections import defaultdict
from typing import List

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        freq = defaultdict(int)
        sum_non_zero = 0
        c_zero = 0
        
        for num in nums:
            if num == 0:
                c_zero += 1
            else:
                freq[num] += 1
                sum_non_zero += num
        
        if sum_non_zero == 0:
            if l <= 0 <= r:
                return (c_zero + 1) % MOD
            else:
                return 0
        
        dp = [0] * (sum_non_zero + 1)
        dp[0] = 1
        
        for x, count in freq.items():
            if x > sum_non_zero:
                continue
            effective_count = min(count, sum_non_zero // x)
            temp_dp = dp.copy()
            
            for r_mod in range(x):
                window_sum = 0
                m_max = (sum_non_zero - r_mod) // x
                for m in range(m_max + 1):
                    s = r_mod + m * x
                    window_sum += dp[s]
                    
                    if m > effective_count:
                        s_prev = s - (effective_count + 1) * x
                        if s_prev >= 0:
                            window_sum -= dp[s_prev]
                    
                    temp_dp[s] = window_sum % MOD
            
            dp = temp_dp
        
        up = min(r, sum_non_zero)
        total = 0
        if up < l:
            total = 0
        else:
            for s in range(max(l, 0), up + 1):
                total = (total + dp[s]) % MOD
        
        if c_zero > 0:
            total = (total * (c_zero + 1)) % MOD
        
        return total