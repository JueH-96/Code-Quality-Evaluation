from collections import deque
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        # We want to count all subarrays where max(nums[i:j+1]) - min(nums[i:j+1]) <= 2
        n = len(nums)
        maxD = deque()  # decreasing deque for maximums
        minD = deque()  # increasing deque for minimums
        left = 0
        count = 0
        
        for right, x in enumerate(nums):
            # Insert x into max-deque (maintain decreasing order)
            while maxD and maxD[-1] < x:
                maxD.pop()
            maxD.append(x)
            
            # Insert x into min-deque (maintain increasing order)
            while minD and minD[-1] > x:
                minD.pop()
            minD.append(x)
            
            # Shrink window from the left while the condition is violated
            # i.e., the current max - min > 2
            while maxD[0] - minD[0] > 2:
                # If the outgoing element equals the front of one of the deques, pop it
                if nums[left] == maxD[0]:
                    maxD.popleft()
                if nums[left] == minD[0]:
                    minD.popleft()
                left += 1
            
            # All subarrays ending at 'right' with start in [left..right] are valid
            count += (right - left + 1)
        
        return count