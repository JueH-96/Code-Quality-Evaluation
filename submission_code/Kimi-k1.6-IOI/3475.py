from collections import deque
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        q = deque()
        operations = 0
        
        for j in range(n):
            # Remove flips that are out of the current window (j-2)
            while q and q[0] < j - 2:
                q.popleft()
            
            # Calculate the current value considering previous flips
            current = nums[j] ^ (len(q) % 2)
            
            if current == 0:
                if j > n - 3:
                    return -1
                q.append(j)
                operations += 1
        
        return operations