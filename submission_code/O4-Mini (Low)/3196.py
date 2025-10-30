from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        # Sort the array to use a sliding window approach
        nums.sort()
        left = 0
        total = 0  # sum of the current window
        max_freq = 1
        
        # Expand the right end of the window
        for right, val in enumerate(nums):
            total += val
            # Compute the cost to make all elements in [left..right] equal to nums[right]:
            # cost = nums[right] * window_size - sum(window)
            window_size = right - left + 1
            cost = val * window_size - total
            
            # If cost exceeds k, shrink the window from the left
            while cost > k:
                total -= nums[left]
                left += 1
                window_size = right - left + 1
                cost = val * window_size - total
            
            # Update the maximum frequency
            max_freq = max(max_freq, window_size)
        
        return max_freq