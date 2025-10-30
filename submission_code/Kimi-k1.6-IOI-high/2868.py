from typing import List
from collections import deque

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        max_deque = deque()
        min_deque = deque()
        left = 0
        result = 0
        
        for right in range(len(nums)):
            # Maintain max_deque in decreasing order
            while max_deque and nums[right] >= nums[max_deque[-1]]:
                max_deque.pop()
            max_deque.append(right)
            
            # Maintain min_deque in increasing order
            while min_deque and nums[right] <= nums[min_deque[-1]]:
                min_deque.pop()
            min_deque.append(right)
            
            # Ensure the window is valid
            while nums[max_deque[0]] - nums[min_deque[0]] > 2:
                left += 1
                # Remove elements out of the window from deques
                while max_deque and max_deque[0] < left:
                    max_deque.popleft()
                while min_deque and min_deque[0] < left:
                    min_deque.popleft()
            
            result += right - left + 1
        
        return result