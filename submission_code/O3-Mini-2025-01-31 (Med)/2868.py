from collections import deque
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        min_deque = deque()
        max_deque = deque()
        left = 0
        count = 0
        
        for right in range(n):
            # Maintain max_deque in decreasing order
            while max_deque and nums[right] > nums[max_deque[-1]]:
                max_deque.pop()
            max_deque.append(right)
            
            # Maintain min_deque in increasing order
            while min_deque and nums[right] < nums[min_deque[-1]]:
                min_deque.pop()
            min_deque.append(right)
            
            # shrink window until condition satisfied: difference between max and min <= 2
            while nums[max_deque[0]] - nums[min_deque[0]] > 2:
                # If left is at the front of either deque, pop it
                if max_deque[0] == left:
                    max_deque.popleft()
                if min_deque[0] == left:
                    min_deque.popleft()
                left += 1
            
            # All subarrays ending at right with start indices between left and right are valid.
            count += (right - left + 1)
        
        return count