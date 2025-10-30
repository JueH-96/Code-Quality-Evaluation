from typing import List
from collections import deque

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        max_deque = deque()
        min_deque = deque()
        left = 0
        count = 0
        
        for right, num in enumerate(nums):
            # Maintain the max_deque
            while max_deque and nums[right] > nums[max_deque[-1]]:
                max_deque.pop()
            max_deque.append(right)
            
            # Maintain the min_deque
            while min_deque and nums[right] < nums[min_deque[-1]]:
                min_deque.pop()
            min_deque.append(right)
            
            # Shrink the window from the left if the condition is violated
            while max_deque and min_deque and nums[max_deque[0]] - nums[min_deque[0]] > 2:
                left += 1
                # Remove indices that are less than left
                while max_deque and max_deque[0] < left:
                    max_deque.popleft()
                while min_deque and min_deque[0] < left:
                    min_deque.popleft()
            
            # Add the number of subarrays in the current window
            count += right - left + 1
        
        return count