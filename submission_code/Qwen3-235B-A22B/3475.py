from collections import deque
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        queue = deque()
        flip_count = 0
        res = 0
        
        for i in range(n):
            # Remove flips that no longer affect the current index i
            while queue and queue[0] <= i - 3:
                queue.popleft()
                flip_count -= 1
            
            current_flips = flip_count % 2
            current_val = nums[i] ^ current_flips
            
            if current_val == 0:
                # Need to flip starting at i
                if i > n - 3:
                    return -1
                queue.append(i)
                flip_count += 1
                res += 1
        
        return res