from collections import defaultdict
from typing import List

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        if k > len(nums):
            return 0
        
        freq = defaultdict(int)
        current_sum = 0
        max_sum = -float('inf')
        
        # Initialize the first window
        for i in range(k):
            num = nums[i]
            freq[num] += 1
            current_sum += num
        
        if len(freq) >= m:
            max_sum = current_sum
        
        # Slide the window through the rest of the array
        for i in range(k, len(nums)):
            left_num = nums[i - k]
            right_num = nums[i]
            
            # Remove the leftmost element from the current window
            freq[left_num] -= 1
            if freq[left_num] == 0:
                del freq[left_num]
            current_sum -= left_num
            
            # Add the new rightmost element
            current_sum += right_num
            freq[right_num] += 1
            
            # Update max_sum if conditions are met
            if len(freq) >= m and current_sum > max_sum:
                max_sum = current_sum
        
        return max_sum if max_sum != -float('inf') else 0