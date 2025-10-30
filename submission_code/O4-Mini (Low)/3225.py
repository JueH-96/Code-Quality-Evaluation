from typing import List

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        freq = {}
        left = 0
        max_len = 0
        
        for right, val in enumerate(nums):
            # Add the current element to the window
            freq[val] = freq.get(val, 0) + 1
            
            # If it violates the "good" condition, shrink from the left
            while freq[val] > k:
                left_val = nums[left]
                freq[left_val] -= 1
                left += 1
            
            # Update maximum length
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        
        return max_len