from collections import deque
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        q = deque()
        count = 0
        n = len(nums)
        for i in range(n):
            # Remove flips that no longer affect the current position
            while q and q[0] <= i - 3:
                q.popleft()
            
            current_flips = len(q) % 2
            current_val = nums[i] ^ current_flips
            
            if current_val == 0:
                # Need to flip here, check if possible
                if i > n - 3:
                    return -1
                count += 1
                q.append(i)
        
        return count