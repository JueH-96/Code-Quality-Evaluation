from collections import deque
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        flip_positions = deque()
        operations = 0
        
        for i in range(n):
            # Remove flips that no longer affect the current index i
            while flip_positions and flip_positions[0] <= i - 3:
                flip_positions.popleft()
            
            current_flips = len(flip_positions)
            current_val = nums[i] ^ (current_flips % 2)
            
            if current_val == 0:
                if i + 2 >= n:
                    return -1
                flip_positions.append(i)
                operations += 1
        
        return operations