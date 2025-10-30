from collections import deque
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        max_deque = deque()
        min_deque = deque()
        left = 0
        total = 0
        
        for right in range(len(nums)):
            # Maintain the max_deque
            while max_deque and nums[right] >= nums[max_deque[-1]]:
                max_deque.pop()
            max_deque.append(right)
            
            # Maintain the min_deque
            while min_deque and nums[right] <= nums[min_deque[-1]]:
                min_deque.pop()
            min_deque.append(right)
            
            # Ensure the window [left, right] is valid
            while nums[max_deque[0]] - nums[min_deque[0]] > 2:
                if max_deque[0] == left:
                    max_deque.popleft()
                if min_deque[0] == left:
                    min_deque.popleft()
                left += 1
            
            # Add the number of valid subarrays ending at 'right'
            total += right - left + 1
        
        return total