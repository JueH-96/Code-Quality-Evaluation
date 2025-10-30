from typing import List
from collections import deque

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        maxd = deque()  # will store indices of elements in decreasing order of their values
        mind = deque()  # will store indices of elements in increasing order of their values
        
        start = 0
        total = 0
        
        for end in range(len(nums)):
            # Maintain maxd in non-increasing order of values
            while maxd and nums[maxd[-1]] <= nums[end]:
                maxd.pop()
            maxd.append(end)
            
            # Maintain mind in non-decreasing order of values
            while mind and nums[mind[-1]] >= nums[end]:
                mind.pop()
            mind.append(end)
            
            # If the current window [start..end] violates the condition max - min <= 2, shrink from the left
            while nums[maxd[0]] - nums[mind[0]] > 2:
                # If the leftmost index in either deque goes out of the window, pop it from the deque
                if maxd[0] == start:
                    maxd.popleft()
                if mind[0] == start:
                    mind.popleft()
                start += 1
            
            # All subarrays ending at 'end' and starting from any index in [start..end] are valid
            total += end - start + 1
        
        return total