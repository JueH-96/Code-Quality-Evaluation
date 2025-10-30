from collections import deque
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        dq = deque()
        current_flips = 0
        operations = 0
        
        for i in range(n):
            # Remove flips that no longer affect the current index
            while dq and dq[0] <= i - 3:
                dq.popleft()
                current_flips -= 1
            
            actual = nums[i] ^ (current_flips % 2)
            if actual == 0:
                if i + 2 >= n:
                    return -1
                operations += 1
                current_flips += 1
                dq.append(i)
        
        return operations