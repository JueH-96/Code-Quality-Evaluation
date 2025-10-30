import collections
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        """
        Calculates the total number of continuous subarrays using a sliding window
        with two deques to efficiently track the window's minimum and maximum.
        """
        
        # A deque to maintain indices of candidates for the window's maximum.
        # The values at these indices will be monotonically decreasing.
        max_deque = collections.deque()
        
        # A deque to maintain indices of candidates for the window's minimum.
        # The values at these indices will be monotonically increasing.
        min_deque = collections.deque()
        
        left = 0
        ans = 0
        n = len(nums)
        
        for right in range(n):
            # Update max_deque: Remove indices of elements smaller than or equal to
            # the current element from the right, then add the current index.
            while max_deque and nums[max_deque[-1]] <= nums[right]:
                max_deque.pop()
            max_deque.append(right)
            
            # Update min_deque: Remove indices of elements larger than or equal to
            # the current element from the right, then add the current index.
            while min_deque and nums[min_deque[-1]] >= nums[right]:
                min_deque.pop()
            min_deque.append(right)
            
            # The window's maximum is at the front of max_deque, and minimum at the front of min_deque.
            # Shrink the window from the left if the condition (max - min <= 2) is violated.
            while nums[max_deque[0]] - nums[min_deque[0]] > 2:
                # The element at 'left' is being removed from the window.
                # If it's the current max or min, pop it from the deques.
                if max_deque[0] == left:
                    max_deque.popleft()
                if min_deque[0] == left:
                    min_deque.popleft()
                
                left += 1
            
            # The window [left, right] is now valid. All subarrays ending at 'right'
            # with a start index >= 'left' are also valid.
            # There are (right - left + 1) such subarrays.
            ans += (right - left + 1)
            
        return ans