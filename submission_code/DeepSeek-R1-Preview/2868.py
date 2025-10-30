from collections import deque
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        max_deque = deque()
        min_deque = deque()
        left = 0
        result = 0
        
        for right in range(len(nums)):
            # Maintain max_deque
            while max_deque and nums[right] >= nums[max_deque[-1]]:
                max_deque.pop()
            max_deque.append(right)
            
            # Maintain min_deque
            while min_deque and nums[right] <= nums[min_deque[-1]]:
                min_deque.pop()
            min_deque.append(right)
            
            # Remove elements from deques that are out of the current window
            while max_deque and max_deque[0] < left:
                max_deque.popleft()
            while min_deque and min_deque[0] < left:
                min_deque.popleft()
            
            # Ensure the current window is valid
            while max_deque and min_deque and nums[max_deque[0]] - nums[min_deque[0]] > 2:
                left += 1
                # Remove elements from deques that are out of the current window
                while max_deque and max_deque[0] < left:
                    max_deque.popleft()
                while min_deque and min_deque[0] < left:
                    min_deque.popleft()
            
            # Add the number of valid subarrays ending at 'right'
            result += right - left + 1
        
        return result