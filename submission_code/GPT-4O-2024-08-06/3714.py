from typing import List

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Sort the array to easily find min and max in subsequences
        nums.sort()
        
        # Precompute powers of 2 up to n
        power_of_2 = [1] * (n + 1)
        for i in range(1, n + 1):
            power_of_2[i] = (power_of_2[i - 1] * 2) % MOD
        
        total_sum = 0
        
        # Calculate contribution of each element as min and max
        for i in range(n):
            # nums[i] as the minimum in subsequences
            if i < n - 1:
                min_contribution = nums[i] * (power_of_2[n - i - 1] - 1) % MOD
            else:
                min_contribution = 0
            
            # nums[i] as the maximum in subsequences
            if i > 0:
                max_contribution = nums[i] * (power_of_2[i] - 1) % MOD
            else:
                max_contribution = 0
            
            # Add contributions to total sum
            total_sum = (total_sum + min_contribution + max_contribution) % MOD
        
        return total_sum