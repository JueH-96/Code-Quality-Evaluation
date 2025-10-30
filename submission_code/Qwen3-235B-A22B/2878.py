from collections import deque
from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        current_ops = 0
        q = deque()
        
        for i in range(n):
            # Remove expired operations
            while q and q[0][0] < i:
                current_ops -= q.popleft()[1]
            
            remaining = nums[i] - current_ops
            if remaining < 0:
                return False
            if remaining > 0:
                if i + k > n:
                    return False
                expiration = i + k - 1
                q.append((expiration, remaining))
                current_ops += remaining
        
        return True