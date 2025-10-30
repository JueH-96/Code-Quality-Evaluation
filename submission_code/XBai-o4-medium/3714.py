from typing import List

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        total = 0
        
        def comb_sum(x, t):
            if t < 0 or x < 0:
                return 0
            if x == 0:
                return 1
            sum_c = 0
            current = 1
            sum_c += current
            for m in range(1, t + 1):
                current = current * (x - m + 1) // m
                sum_c += current
            return sum_c
        
        for i in range(n):
            max_t = min(k - 1, i)
            sum_max = comb_sum(i, max_t)
            
            r = (n - 1 - i)
            min_t = min(k - 1, r)
            sum_min = comb_sum(r, min_t)
            
            total += (sum_max + sum_min) * nums[i]
        
        return total % MOD