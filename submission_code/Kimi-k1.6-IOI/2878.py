from collections import deque
from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        max_start = n - k
        q = deque()
        current = 0
        
        for i in range(n):
            # Remove operations that no longer affect the current index
            while q and q[0][0] + k - 1 < i:
                removed = q.popleft()
                current -= removed[1]
            
            needed = nums[i] - current
            if needed < 0:
                return False
            
            if i > max_start:
                if needed != 0:
                    return False
            else:
                if needed > 0:
                    q.append((i, needed))
                    current += needed
        
        return True