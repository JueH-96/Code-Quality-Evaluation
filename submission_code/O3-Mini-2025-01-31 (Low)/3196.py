from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        left = 0
        total = 0
        res = 0
        
        for right in range(len(nums)):
            total += nums[right]
            # Compute the cost to turn all numbers in window to nums[right]
            # cost = nums[right] * window_size - (sum of window)
            while (nums[right] * (right - left + 1) - total) > k:
                total -= nums[left]
                left += 1
            res = max(res, right - left + 1)
        return res