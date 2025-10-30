from sortedcontainers import SortedList
from typing import List

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        sl = SortedList()
        min_diff = float('inf')
        
        for i in range(x, n):
            sl.add(nums[i - x])
            pos = sl.bisect_left(nums[i])
            
            if pos < len(sl):
                min_diff = min(min_diff, sl[pos] - nums[i])
            if pos > 0:
                min_diff = min(min_diff, nums[i] - sl[pos - 1])
        
        return min_diff