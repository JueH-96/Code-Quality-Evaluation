from collections import Counter
from typing import List

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        max_sum = 0
        current_sum = 0
        element_count = Counter()
        
        for i in range(len(nums)):
            # Add the current element to the window
            element_count[nums[i]] += 1
            current_sum += nums[i]
            
            # Remove the element that is left out of the window
            if i >= k:
                element_count[nums[i - k]] -= 1
                current_sum -= nums[i - k]
                if element_count[nums[i - k]] == 0:
                    del element_count[nums[i - k]]
            
            # Check if the current window is almost unique
            if len(element_count) >= m:
                max_sum = max(max_sum, current_sum)
        
        return max_sum