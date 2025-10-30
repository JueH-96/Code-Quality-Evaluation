from collections import deque
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        left = 0
        total = 0
        
        # Deques to maintain the indices for min and max in current window
        min_dq = deque()
        max_dq = deque()
        
        for right in range(n):
            # Maintain max_dq - decreasing order, so the first element is the maximum
            while max_dq and nums[right] > nums[max_dq[-1]]:
                max_dq.pop()
            max_dq.append(right)
            
            # Maintain min_dq - increasing order, so the first element is the minimum
            while min_dq and nums[right] < nums[min_dq[-1]]:
                min_dq.pop()
            min_dq.append(right)
            
            # Ensure that the difference between maximum and minimum in the window is <= 2.
            # If not, move left pointer to shrink the window.
            while nums[max_dq[0]] - nums[min_dq[0]] > 2:
                if max_dq[0] == left:
                    max_dq.popleft()
                if min_dq[0] == left:
                    min_dq.popleft()
                left += 1
            
            # All subarrays ending at 'right' with start index in [left, right] are valid.
            total += (right - left + 1)
        
        return total