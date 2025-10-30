from collections import deque
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        max_deque = deque()
        min_deque = deque()
        left = 0
        total = 0
        
        for right in range(len(nums)):
            # Update max_deque: remove elements smaller than current value from the end
            while max_deque and nums[right] >= nums[max_deque[-1]]:
                max_deque.pop()
            max_deque.append(right)
            
            # Update min_deque: remove elements larger than current value from the end
            while min_deque and nums[right] <= nums[min_deque[-1]]:
                min_deque.pop()
            min_deque.append(right)
            
            # Ensure the front elements are within the current window [left, right]
            while max_deque[0] < left:
                max_deque.popleft()
            while min_deque[0] < left:
                min_deque.popleft()
            
            # Adjust left until the window is valid
            while nums[max_deque[0]] - nums[min_deque[0]] > 2:
                left += 1
                # After moving left, check if front elements are still within the window
                while max_deque[0] < left:
                    max_deque.popleft()
                while min_deque[0] < left:
                    min_deque.popleft()
            
            # Add the number of valid subarrays ending at 'right'
            total += right - left + 1
        
        return total