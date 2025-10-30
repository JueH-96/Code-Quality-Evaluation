from typing import List
from collections import deque

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        current_operations = 0
        operations_queue = deque()
        
        for i in range(len(nums)):
            # Remove operations that no longer affect position i
            if operations_queue and operations_queue[0][0] <= i:
                current_operations -= operations_queue[0][1]
                operations_queue.popleft()
            
            remaining = nums[i] - current_operations
            if remaining < 0:
                return False
            if remaining > 0:
                if i + k > len(nums):
                    return False
                # Apply 'remaining' operations starting at i
                current_operations += remaining
                # These operations will end at i + k
                operations_queue.append((i + k, remaining))
        
        return True