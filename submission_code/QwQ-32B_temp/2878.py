from typing import List
from collections import deque

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        q = deque()
        current_ops = 0
        
        for i in range(n):
            # Remove operations that no longer affect the current index
            while q and q[0][0] <= i:
                end, cnt = q.popleft()
                current_ops -= cnt
            
            delta = nums[i] - current_ops
            if delta < 0:
                return False
            if delta == 0:
                continue
            
            # Check if we can start a new operation here
            if i + k > n:
                return False
            
            current_ops += delta
            q.append((i + k, delta))
        
        return True