from typing import List
import bisect

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        sorted_list = []
        min_diff = float('inf')
        
        for i in range(x, n):
            bisect.insort(sorted_list, nums[i - x])
            pos = bisect.bisect_left(sorted_list, nums[i])
            
            if pos < len(sorted_list):
                min_diff = min(min_diff, abs(nums[i] - sorted_list[pos]))
            if pos > 0:
                min_diff = min(min_diff, abs(nums[i] - sorted_list[pos - 1]))
        
        return min_diff