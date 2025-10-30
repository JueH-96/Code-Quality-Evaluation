from collections import deque
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        max_deque = deque()
        min_deque = deque()
        left = 0
        result = 0
        
        for right in range(len(nums)):
            # Update max_deque with the new element's index
            while max_deque and nums[right] >= nums[max_deque[-1]]:
                max_deque.pop()
            max_deque.append(right)
            
            # Update min_deque with the new element's index
            while min_deque and nums[right] <= nums[min_deque[-1]]:
                min_deque.pop()
            min_deque.append(right)
            
            # Ensure the window [left, right] is valid
            while nums[max_deque[0]] - nums[min_deque[0]] > 2:
                left += 1
                # Remove elements out of the window from the deques
                if max_deque[0] < left:
                    max_deque.popleft()
                if min_deque[0] < left:
                    min_deque.popleft()
            
            # Add the number of valid subarrays ending at 'right'
            result += right - left + 1
        
        return result