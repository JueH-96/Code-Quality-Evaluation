from collections import deque
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        total = 0
        left = 0
        max_deque = deque()
        min_deque = deque()
        
        for right in range(len(nums)):
            # Maintain max_deque for maximum values in the current window
            while max_deque and nums[right] >= nums[max_deque[-1]]:
                max_deque.pop()
            max_deque.append(right)
            
            # Maintain min_deque for minimum values in the current window
            while min_deque and nums[right] <= nums[min_deque[-1]]:
                min_deque.pop()
            min_deque.append(right)
            
            # Adjust left boundary to ensure the window is valid
            while True:
                # Remove indices out of the current window from deques
                while max_deque and max_deque[0] < left:
                    max_deque.popleft()
                while min_deque and min_deque[0] < left:
                    min_deque.popleft()
                
                current_max = nums[max_deque[0]]
                current_min = nums[min_deque[0]]
                
                if current_max - current_min <= 2:
                    break
                else:
                    left += 1
            
            # Add the number of valid subarrays ending at current right
            total += (right - left + 1)
        
        return total