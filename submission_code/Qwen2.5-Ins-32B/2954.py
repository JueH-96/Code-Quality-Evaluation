from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if n < k:
            return 0
        
        # Initialize the sliding window
        window = defaultdict(int)
        for i in range(k):
            window[nums[i]] += 1
        
        max_sum = 0
        current_sum = sum(nums[:k])
        
        if len(window) >= m:
            max_sum = current_sum
        
        # Slide the window
        for i in range(k, n):
            window[nums[i]] += 1
            window[nums[i - k]] -= 1
            if window[nums[i - k]] == 0:
                del window[nums[i - k]]
            
            current_sum += nums[i] - nums[i - k]
            
            if len(window) >= m:
                max_sum = max(max_sum, current_sum)
        
        return max_sum