from typing import List
from collections import defaultdict

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        left = 0
        max_freq = 0
        window_counter = defaultdict(int)
        
        for right in range(len(nums)):
            window_counter[nums[right]] += 1
            window_size = right - left + 1
            
            # Check if the window needs to be adjusted
            while nums[right] - nums[left] > k or (window_size - window_counter[nums[right]]) > numOperations:
                window_counter[nums[left]] -= 1
                if window_counter[nums[left]] == 0:
                    del window_counter[nums[left]]
                left += 1
                window_size -= 1
            
            # Update the maximum frequency
            max_freq = max(max_freq, window_size)
        
        return max_freq