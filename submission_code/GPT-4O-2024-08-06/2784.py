from typing import List

class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        
        # Precompute powers of 2 up to n
        power_of_2 = [1] * n
        for i in range(1, n):
            power_of_2[i] = (power_of_2[i - 1] * 2) % MOD
        
        sum_power = 0
        for i in range(n):
            max_val = nums[i]
            # Calculate contribution of nums[i] as the maximum in all subsets where it is the maximum
            # It can be the maximum in 2^i subsets (all subsets of elements before it)
            contribution = (max_val * max_val % MOD) * nums[i] % MOD * power_of_2[i] % MOD
            sum_power = (sum_power + contribution) % MOD
        
        return sum_power