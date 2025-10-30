from collections import deque
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        dq = deque()
        operations = 0
        
        for i in range(n):
            # Remove flips that no longer affect the current index i
            while dq and dq[0] + 2 < i:
                dq.popleft()
            
            current_flips = len(dq)
            effective_value = (nums[i] + current_flips) % 2
            
            if effective_value == 0:
                if i + 2 >= n:
                    return -1
                dq.append(i)
                operations += 1
        
        return operations