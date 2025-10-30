from typing import List
from collections import defaultdict

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # counts[val] = number of windows (size k) that contain val
        counts = defaultdict(int)

        for start in range(n - k + 1):
            # elements present in the current window
            window_vals = set(nums[start:start + k])
            for val in window_vals:
                counts[val] += 1

        # collect values that occur in exactly one window
        candidates = [val for val, c in counts.items() if c == 1]
        return max(candidates) if candidates else -1