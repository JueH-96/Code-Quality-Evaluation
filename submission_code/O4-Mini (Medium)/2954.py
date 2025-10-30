from typing import List

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        """
        Returns the maximum sum of any subarray of length k that contains
        at least m distinct elements. If no such subarray exists, returns 0.
        """
        n = len(nums)
        # If there aren't enough elements for one window, return 0
        if n < k:
            return 0
        
        from collections import defaultdict
        
        count = defaultdict(int)
        distinct = 0
        curr_sum = 0
        max_sum = 0
        
        # Initialize the first window [0..k-1]
        for i in range(k):
            val = nums[i]
            curr_sum += val
            if count[val] == 0:
                distinct += 1
            count[val] += 1
        
        # Check the first window
        if distinct >= m:
            max_sum = curr_sum
        
        # Slide the window from i=k to n-1
        for i in range(k, n):
            # Add the new element nums[i]
            add_val = nums[i]
            curr_sum += add_val
            if count[add_val] == 0:
                distinct += 1
            count[add_val] += 1
            
            # Remove the element going out of the window nums[i-k]
            rem_val = nums[i - k]
            count[rem_val] -= 1
            curr_sum -= rem_val
            if count[rem_val] == 0:
                distinct -= 1
                del count[rem_val]
            
            # Check if this window is "almost unique"
            if distinct >= m:
                if curr_sum > max_sum:
                    max_sum = curr_sum
        
        return max_sum