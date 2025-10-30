from typing import List
from collections import defaultdict

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        left = 0
        max_length = 0
        freq = defaultdict(int)
        
        for right in range(len(nums)):
            freq[nums[right]] += 1
            
            # If the frequency of any element exceeds k, move the left pointer
            while freq[nums[right]] > k:
                freq[nums[left]] -= 1
                left += 1
            
            # Calculate the length of the current good subarray
            max_length = max(max_length, right - left + 1)
        
        return max_length