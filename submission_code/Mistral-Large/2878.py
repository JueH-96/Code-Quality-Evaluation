from typing import List
from collections import deque

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        max_val = max(nums)

        # Sliding window to maintain the maximum and minimum values in the current subarray of size k
        window = deque(nums[:k])
        current_max = max(window)
        current_min = min(window)

        for i in range(k, n):
            window.popleft()
            window.append(nums[i])
            current_max = max(current_max, nums[i])
            current_min = min(current_min, nums[i])

            if current_max - current_min > max_val:
                return False

        return True