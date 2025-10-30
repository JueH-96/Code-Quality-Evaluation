from typing import List
from collections import deque

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        max_q = deque()
        min_q = deque()
        left = 0
        result = 0
        
        for right in range(len(nums)):
            # Maintain max_q
            while max_q and nums[right] > max_q[-1]:
                max_q.pop()
            max_q.append(nums[right])
            
            # Maintain min_q
            while min_q and nums[right] < min_q[-1]:
                min_q.pop()
            min_q.append(nums[right])
            
            # Shrink the window if the condition is violated
            while max_q[0] - min_q[0] > 2:
                if nums[left] == max_q[0]:
                    max_q.popleft()
                if nums[left] == min_q[0]:
                    min_q.popleft()
                left += 1
            
            # Add the number of valid subarrays ending at 'right'
            result += right - left + 1
        
        return result