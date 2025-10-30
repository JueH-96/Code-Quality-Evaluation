from collections import deque
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        max_deque = deque()
        min_deque = deque()
        left = 0
        ans = 0
        for right in range(len(nums)):
            # Maintain max_deque in decreasing order
            while max_deque and nums[max_deque[-1]] < nums[right]:
                max_deque.pop()
            max_deque.append(right)
            
            # Maintain min_deque in increasing order
            while min_deque and nums[min_deque[-1]] > nums[right]:
                min_deque.pop()
            min_deque.append(right)
            
            # Shrink the window from left if max - min > 2
            while nums[max_deque[0]] - nums[min_deque[0]] > 2:
                left += 1
                # Remove elements out of the window from max_deque
                while max_deque and max_deque[0] < left:
                    max_deque.popleft()
                # Remove elements out of the window from min_deque
                while min_deque and min_deque[0] < left:
                    min_deque.popleft()
            
            # Add the number of valid subarrays ending at right
            ans += (right - left + 1)
        
        return ans