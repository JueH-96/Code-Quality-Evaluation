from collections import deque
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        # deques to maintain the minimum and maximum in the current sliding window
        min_deque = deque()  # increasing
        max_deque = deque()  # decreasing
        
        left = 0
        count = 0
        
        for right in range(n):
            # maintain deque for minimum values: increasing order
            while min_deque and nums[min_deque[-1]] > nums[right]:
                min_deque.pop()
            min_deque.append(right)
            
            # maintain deque for maximum values: decreasing order
            while max_deque and nums[max_deque[-1]] < nums[right]:
                max_deque.pop()
            max_deque.append(right)
            
            # adjust window such that max - min <= 2
            while nums[max_deque[0]] - nums[min_deque[0]] > 2:
                # move left pointer
                if min_deque[0] == left:
                    min_deque.popleft()
                if max_deque[0] == left:
                    max_deque.popleft()
                left += 1
                
            # all subarrays ending at right with start index in [left, right] are valid
            count += (right - left + 1)
            
        return count