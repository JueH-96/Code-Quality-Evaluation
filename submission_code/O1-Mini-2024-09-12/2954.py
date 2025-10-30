from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        if len(nums) < k:
            return 0
        
        freq = defaultdict(int)
        current_sum = 0
        unique = 0
        max_sum = 0
        found = False
        
        # Initialize the first window
        for i in range(k):
            num = nums[i]
            current_sum += num
            freq[num] += 1
            if freq[num] == 1:
                unique += 1
        
        if unique >= m:
            max_sum = current_sum
            found = True
        
        # Slide the window
        for i in range(k, len(nums)):
            # Remove the element exiting the window
            exit_num = nums[i - k]
            current_sum -= exit_num
            freq[exit_num] -= 1
            if freq[exit_num] == 0:
                unique -= 1
            
            # Add the new element entering the window
            enter_num = nums[i]
            current_sum += enter_num
            freq[enter_num] += 1
            if freq[enter_num] == 1:
                unique += 1
            
            # Check if current window is almost unique
            if unique >= m:
                if not found or current_sum > max_sum:
                    max_sum = current_sum
                    found = True
        
        return max_sum if found else 0