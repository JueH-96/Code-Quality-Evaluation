from typing import List
from collections import deque

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        max_deque = deque()
        min_deque = deque()
        left = 0
        total = 0
        
        for right in range(len(nums)):
            # Update max_deque to maintain decreasing order
            while max_deque and nums[max_deque[-1]] <= nums[right]:
                max_deque.pop()
            max_deque.append(right)
            
            # Update min_deque to maintain increasing order
            while min_deque and nums[min_deque[-1]] >= nums[right]:
                min_deque.pop()
            min_deque.append(right)
            
            # Ensure the window [left, right] is valid
            while nums[max_deque[0]] - nums[min_deque[0]] > 2:
                left += 1
                # Remove elements from deques that are out of the window
                while max_deque and max_deque[0] < left:
                    max_deque.popleft()
                while min_deque and min_deque[0] < left:
                    min_deque.popleft()
            
            # Add the number of valid subarrays ending at 'right'
            total += right - left + 1
        
        return total