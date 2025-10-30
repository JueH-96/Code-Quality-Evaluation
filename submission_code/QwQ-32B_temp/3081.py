from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return n
        
        max_freq = 1
        current = 1
        
        for i in range(1, n):
            if nums[i] == nums[i-1]:
                current += 1
                if current > max_freq:
                    max_freq = current
            else:
                current = 1
        
        return max(2 * max_freq - n, n % 2)