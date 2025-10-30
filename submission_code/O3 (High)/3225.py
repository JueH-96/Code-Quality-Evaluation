from typing import List
from collections import defaultdict

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        # frequency counter for the current window
        freq = defaultdict(int)
        left = 0
        best = 0
        
        for right, val in enumerate(nums):
            freq[val] += 1          # include nums[right] in the window
            
            # if the current value appears too often, shrink the window
            while freq[val] > k:
                freq[nums[left]] -= 1
                left += 1           # move the left boundary rightwards
            
            # all counts are now ≤ k
            best = max(best, right - left + 1)
        
        return best