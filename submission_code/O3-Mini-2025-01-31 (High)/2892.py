from typing import List

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        # Determine candidate n which should be the maximum in nums
        n = max(nums)
        # The base array should have n + 1 elements: [1, 2, ..., n-1, n, n]
        if len(nums) != n + 1:
            return False
        
        # Count the frequency of each element in nums
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
            
        # For all numbers from 1 to n-1, the frequency should be exactly 1.
        for i in range(1, n):
            if freq.get(i, 0) != 1:
                return False
        
        # The maximum element n should occur exactly twice.
        if freq.get(n, 0) != 2:
            return False
        
        return True