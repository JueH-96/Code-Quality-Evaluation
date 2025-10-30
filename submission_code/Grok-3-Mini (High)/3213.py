from typing import List
import math

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        # Find the maximum value in nums
        max_val = max(nums)
        
        # Initialize variables for sliding window
        left = 0
        count = 0  # Count of max_val in the current window
        sum_less = 0  # Sum of subarrays with fewer than k max_val
        N = len(nums)
        
        # Slide the right pointer
        for right in range(N):
            # Add the element at right to the window
            if nums[right] == max_val:
                count += 1
            
            # Shrink the window from the left until count < k
            while count >= k and left <= right:
                if nums[left] == max_val:
                    count -= 1
                left += 1  # Move left pointer after removing the element
            
            # After adjustment, add the number of subarrays ending at right with count < k
            # This is the size of the current window [left, right]
            sum_less += max(0, right - left + 1)
        
        # Total number of subarrays
        total = N * (N + 1) // 2
        
        # Number of subarrays with at least k max_val is total minus those with fewer than k
        return total - sum_less