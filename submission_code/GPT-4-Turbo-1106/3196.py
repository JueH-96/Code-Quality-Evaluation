from typing import List
from collections import Counter

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        left = 0
        max_freq = 0
        total = 0
        
        for right in range(len(nums)):
            total += nums[right]
            
            # Check if we can make all elements from left to right the same as nums[right]
            while nums[right] * (right - left + 1) > total + k:
                total -= nums[left]
                left += 1
            
            max_freq = max(max_freq, right - left + 1)
        
        return max_freq

# Example usage:
# sol = Solution()
# print(sol.maxFrequencyScore([1,2,6,4], 3))  # Output: 3
# print(sol.maxFrequencyScore([1,4,4,2,4], 0))  # Output: 3