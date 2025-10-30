from typing import List
import bisect

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        sorted_list = []
        n = len(nums)
        min_diff = float('inf')
        
        for i in range(n):
            if i >= x:
                # Add nums[i - x] to the sorted list
                bisect.insort(sorted_list, nums[i - x])
            if sorted_list:
                # Find the position to insert nums[i]
                pos = bisect.bisect_left(sorted_list, nums[i])
                # Check the closest values
                if pos > 0:
                    min_diff = min(min_diff, abs(nums[i] - sorted_list[pos - 1]))
                if pos < len(sorted_list):
                    min_diff = min(min_diff, abs(nums[i] - sorted_list[pos]))
        return min_diff