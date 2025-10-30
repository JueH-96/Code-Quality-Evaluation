from typing import List
from math import comb

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Sort the array to easily access min and max elements
        nums.sort()
        
        total_sum = 0
        
        # Calculate contributions of each element as min and max
        for i in range(n):
            # Calculate how many times nums[i] will be the minimum
            # It will be the minimum in all combinations where it is chosen
            # and we can choose (k-1) elements from the remaining (n-i-1) elements
            if k > 1:
                min_contrib = comb(n - i - 1, k - 1) if n - i - 1 >= k - 1 else 0
            else:
                min_contrib = 1
            
            # Calculate how many times nums[i] will be the maximum
            # It will be the maximum in all combinations where it is chosen
            # and we can choose (k-1) elements from the remaining (i) elements
            if k > 1:
                max_contrib = comb(i, k - 1) if i >= k - 1 else 0
            else:
                max_contrib = 1
            
            total_sum += (min_contrib * nums[i] + max_contrib * nums[i]) % MOD
            total_sum %= MOD
        
        return total_sum