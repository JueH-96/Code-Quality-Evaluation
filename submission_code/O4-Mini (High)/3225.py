from typing import List

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        counts = {}
        left = 0
        max_len = 0
        
        for right, x in enumerate(nums):
            counts[x] = counts.get(x, 0) + 1
            
            # If the count of x exceeds k, shrink from the left
            while counts[x] > k:
                counts[nums[left]] -= 1
                left += 1
            
            # Update the maximum valid window size
            max_len = max(max_len, right - left + 1)
        
        return max_len