import bisect
from typing import List

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        if x == 0:
            nums_sorted = sorted(nums)
            min_diff = float('inf')
            for i in range(len(nums_sorted) - 1):
                diff = nums_sorted[i + 1] - nums_sorted[i]
                if diff < min_diff:
                    min_diff = diff
            return min_diff
        
        sorted_left = []
        min_diff = float('inf')
        n = len(nums)
        
        # Left pass: elements at least x indices before i
        for i in range(n):
            if i >= x:
                bisect.insort(sorted_left, nums[i - x])
            if sorted_left:
                pos = bisect.bisect_left(sorted_left, nums[i])
                if pos < len(sorted_left):
                    min_diff = min(min_diff, abs(sorted_left[pos] - nums[i]))
                if pos > 0:
                    min_diff = min(min_diff, abs(sorted_left[pos - 1] - nums[i]))
        
        # Right pass: elements at least x indices after i
        sorted_right = []
        for i in range(n - 1, -1, -1):
            if i + x < n:
                bisect.insort(sorted_right, nums[i + x])
            if sorted_right:
                pos = bisect.bisect_left(sorted_right, nums[i])
                if pos < len(sorted_right):
                    min_diff = min(min_diff, abs(sorted_right[pos] - nums[i]))
                if pos > 0:
                    min_diff = min(min_diff, abs(sorted_right[pos - 1] - nums[i]))
        
        return min_diff