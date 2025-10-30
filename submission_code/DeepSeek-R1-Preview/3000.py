from typing import List
import bisect

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        min_diff = float('inf')
        n = len(nums)
        
        # First pass: left to right, considering j <= i - x
        sorted_list = []
        for i in range(n):
            if i >= x:
                # Add the element x positions before current i
                bisect.insort(sorted_list, nums[i - x])
            if sorted_list:
                current = nums[i]
                pos = bisect.bisect_left(sorted_list, current)
                # Check the closest elements around the insertion point
                if pos > 0:
                    min_diff = min(min_diff, abs(current - sorted_list[pos - 1]))
                if pos < len(sorted_list):
                    min_diff = min(min_diff, abs(current - sorted_list[pos]))
        
        # Second pass: right to left, considering j >= i + x
        sorted_list2 = []
        for i in range(n - 1, -1, -1):
            if i + x < n:
                # Add the element x positions after current i
                bisect.insort(sorted_list2, nums[i + x])
            if sorted_list2:
                current = nums[i]
                pos = bisect.bisect_left(sorted_list2, current)
                # Check the closest elements around the insertion point
                if pos > 0:
                    min_diff = min(min_diff, abs(current - sorted_list2[pos - 1]))
                if pos < len(sorted_list2):
                    min_diff = min(min_diff, abs(current - sorted_list2[pos]))
        
        return min_diff