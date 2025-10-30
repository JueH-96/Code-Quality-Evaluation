from collections import deque
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        # Deques to maintain the current window's max and min
        max_d = deque()
        min_d = deque()
        l = 0
        ans = 0
        
        for r in range(n):
            x = nums[r]
            # Maintain max deque (monotonically decreasing)
            while max_d and max_d[-1] < x:
                max_d.pop()
            max_d.append(x)
            
            # Maintain min deque (monotonically increasing)
            while min_d and min_d[-1] > x:
                min_d.pop()
            min_d.append(x)
            
            # Shrink window from left if invalid (max - min > 2)
            while max_d[0] - min_d[0] > 2:
                # Pop from deques if the outgoing element equals their front
                if nums[l] == max_d[0]:
                    max_d.popleft()
                if nums[l] == min_d[0]:
                    min_d.popleft()
                l += 1
            
            # All subarrays ending at r with start in [l..r] are valid
            ans += (r - l + 1)
        
        return ans