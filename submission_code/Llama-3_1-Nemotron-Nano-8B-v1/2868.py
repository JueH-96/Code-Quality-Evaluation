from collections import deque
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        max_deque = deque()
        min_deque = deque()
        left = 0
        result = 0
        for right in range(len(nums)):
            # Maintain max_deque (elements in decreasing order)
            while max_deque and nums[right] >= max_deque[-1][0]:
                max_deque.pop()
            max_deque.append((nums[right], right))
            # Maintain min_deque (elements in increasing order)
            while min_deque and nums[right] <= min_deque[-1][0]:
                min_deque.pop()
            min_deque.append((nums[right], right))
            
            # Ensure the window [left, right] is valid
            while True:
                current_max = max_deque[0][0]
                current_min = min_deque[0][0]
                if current_max - current_min <= 2:
                    break
                # Move left forward
                left += 1
                # Remove elements out of the window from deques
                while max_deque and max_deque[0][1] < left:
                    max_deque.popleft()
                while min_deque and min_deque[0][1] < left:
                    min_deque.popleft()
                # Re-check after adjusting deques
                if not max_deque or not min_deque:
                    break  # This case should not occur as nums is non-empty
            # Add the count of valid subarrays ending at 'right'
            result += right - left + 1
        return result