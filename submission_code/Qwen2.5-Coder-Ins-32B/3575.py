from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        max_value = 0
        n = len(nums)
        
        # Iterate over all possible starting points of the subsequence
        for i in range(n - 2 * k + 1):
            # Calculate the OR for the first half
            left_or = 0
            for j in range(i, i + k):
                left_or |= nums[j]
            
            # Calculate the OR for the second half
            right_or = 0
            for j in range(i + k, i + 2 * k):
                right_or |= nums[j]
            
            # Calculate the value of the sequence
            value = left_or ^ right_or
            
            # Update the maximum value
            max_value = max(max_value, value)
        
        return max_value