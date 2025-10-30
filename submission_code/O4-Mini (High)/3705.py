from typing import List
from collections import defaultdict

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        # counts[x] will track the number of size-k subarrays in which x appears
        counts = defaultdict(int)
        n = len(nums)
        
        # Slide a window of length k over nums
        for i in range(n - k + 1):
            window_vals = set(nums[i:i + k])
            for x in window_vals:
                counts[x] += 1
        
        # Find the maximum x with exactly one appearance
        ans = -1
        for x, cnt in counts.items():
            if cnt == 1 and x > ans:
                ans = x
        
        return ans