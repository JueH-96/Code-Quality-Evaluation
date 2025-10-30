from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        if k > len(nums):
            return 0
        
        freq = defaultdict(int)
        sum_current = 0
        distinct = 0
        
        # Initialize the frequency dictionary and sum with the first k elements
        for i in range(k):
            freq[nums[i]] += 1
            sum_current += nums[i]
        
        # Determine the number of distinct elements in the initial window
        distinct = len(freq)
        
        # Initialize max_sum
        max_sum = sum_current if distinct >= m else 0
        
        # Slide the window from k to len(nums) - 1
        for i in range(k, len(nums)):
            # Remove the element going out of the window
            sum_current -= nums[i - k]
            freq[nums[i - k]] -= 1
            if freq[nums[i - k]] == 0:
                distinct -= 1
                del freq[nums[i - k]]  # Optional: remove the key to keep the dictionary clean
        
            # Add the new element to the window
            sum_current += nums[i]
            if freq[nums[i]] == 0:
                distinct += 1
            freq[nums[i]] += 1
        
            # Check if the current window meets the criteria
            if distinct >= m:
                if sum_current > max_sum:
                    max_sum = sum_current
        
        return max_sum