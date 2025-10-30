from typing import List
from collections import deque

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        max_d = deque()
        min_d = deque()
        left = 0
        total = 0
        for right in range(len(nums)):
            # Update max deque
            while max_d and nums[right] > nums[max_d[-1]]:
                max_d.pop()
            max_d.append(right)
            
            # Update min deque
            while min_d and nums[right] < nums[min_d[-1]]:
                min_d.pop()
            min_d.append(right)
            
            # Check if current window is valid
            while nums[max_d[0]] - nums[min_d[0]] > 2:
                # Move left
                if max_d[0] == left:
                    max_d.popleft()
                if min_d[0] == left:
                    min_d.popleft()
                left += 1
            
            # Add the number of subarrays ending at right
            total += right - left + 1
        return total