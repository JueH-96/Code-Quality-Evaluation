from collections import deque
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        """
        Count the number of subarrays where the difference between the maximum
        and minimum elements in the subarray is at most 2.
        We use a sliding window with two monotonic deques to track the current
        window's max and min in O(1) time per update.
        """
        n = len(nums)
        # Deques to maintain indices of potential min and max
        min_deque = deque()  # increasing values
        max_deque = deque()  # decreasing values
        
        total = 0
        left = 0
        
        for right, val in enumerate(nums):
            # Insert into min_deque (keep increasing)
            while min_deque and nums[min_deque[-1]] > val:
                min_deque.pop()
            min_deque.append(right)
            
            # Insert into max_deque (keep decreasing)
            while max_deque and nums[max_deque[-1]] < val:
                max_deque.pop()
            max_deque.append(right)
            
            # Shrink window from the left until it's valid again
            while nums[max_deque[0]] - nums[min_deque[0]] > 2:
                if min_deque[0] == left:
                    min_deque.popleft()
                if max_deque[0] == left:
                    max_deque.popleft()
                left += 1
            
            # All subarrays ending at `right` with start in [left..right] are valid
            total += (right - left + 1)
        
        return total