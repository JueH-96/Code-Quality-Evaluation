from typing import List
from collections import deque

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        dq = deque(nums)
        if len(dq) < 2:
            return 0  # According to constraints, nums.length >= 2, so this may not be necessary
        initial_sum = dq[0] + dq[1]
        count = 0
        while len(dq) >= 2:
            if dq[0] + dq[1] == initial_sum:
                dq.popleft()
                dq.popleft()
                count += 1
            else:
                break
        return count