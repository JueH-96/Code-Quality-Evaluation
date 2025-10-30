from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if k > n:
            return 0
        
        count = defaultdict(int)
        current_sum = 0
        max_sum = 0
        
        # Initialize first window of size k
        for i in range(k):
            count[nums[i]] += 1
            current_sum += nums[i]
        
        if len(count) >= m:
            max_sum = current_sum
        
        # Slide the window from index k to n-1
        for i in range(k, n):
            # Add new element to the window
            count[nums[i]] += 1
            current_sum += nums[i]
            
            # Remove element that goes out of the window
            left_val = nums[i - k]
            count[left_val] -= 1
            if count[left_val] == 0:
                del count[left_val]
            current_sum -= left_val
            
            # Check the distinct count condition
            if len(count) >= m:
                max_sum = max(max_sum, current_sum)
                
        return max_sum