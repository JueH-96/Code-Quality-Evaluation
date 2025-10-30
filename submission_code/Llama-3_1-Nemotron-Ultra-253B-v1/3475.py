from typing import List
from collections import deque

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        queue = deque()
        current_flips = 0
        for i in range(n):
            # Remove expired flips
            while queue and queue[0] <= i:
                queue.popleft()
                current_flips -= 1
            # Compute current value after flips
            current_val = nums[i]
            if current_flips % 2 == 1:
                current_val ^= 1
            if current_val == 0:
                # Check if flipping is possible
                if i + 2 >= n:
                    return -1
                res += 1
                current_flips += 1
                queue.append(i + 3)
        return res