from typing import List
import collections

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if k > n:
            return 0
        
        count = collections.Counter()
        current_sum = 0
        max_sum = 0
        
        # Initialize the first window
        for i in range(k):
            current_sum += nums[i]
            count[nums[i]] += 1
        
        if len(count) >= m:
            max_sum = current_sum
        
        # Slide the window
        for i in range(k, n):
            # Remove the element going out of the window
            out_elem = nums[i - k]
            count[out_elem] -= 1
            if count[out_elem] == 0:
                del count[out_elem]
            current_sum -= out_elem
            
            # Add the new element
            in_elem = nums[i]
            count[in_elem] += 1
            current_sum += in_elem
            
            # Check distinct count
            if len(count) >= m:
                max_sum = max(max_sum, current_sum)
        
        return max_sum