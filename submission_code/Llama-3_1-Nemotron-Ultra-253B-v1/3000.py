from typing import List
import bisect

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        left_sorted = []
        right_sorted = []
        min_diff = float('inf')
        
        # Initialize right_sorted with elements from x to n-1
        for j in range(x, n):
            bisect.insort(right_sorted, nums[j])
        
        for i in range(n):
            # Add element to left_sorted if possible
            if i >= x:
                bisect.insort(left_sorted, nums[i - x])
            
            # Check left_sorted
            if left_sorted:
                idx = bisect.bisect_left(left_sorted, nums[i])
                if idx < len(left_sorted):
                    min_diff = min(min_diff, abs(left_sorted[idx] - nums[i]))
                if idx > 0:
                    min_diff = min(min_diff, abs(left_sorted[idx-1] - nums[i]))
            
            # Check right_sorted
            if right_sorted:
                idx = bisect.bisect_left(right_sorted, nums[i])
                if idx < len(right_sorted):
                    min_diff = min(min_diff, abs(right_sorted[idx] - nums[i]))
                if idx > 0:
                    min_diff = min(min_diff, abs(right_sorted[idx-1] - nums[i]))
            
            # Remove element from right_sorted if possible
            if i < x:
                j = i + x
                if j < n:
                    # Find the index of nums[j] in right_sorted and remove it
                    idx = bisect.bisect_left(right_sorted, nums[j])
                    if idx < len(right_sorted) and right_sorted[idx] == nums[j]:
                        right_sorted.pop(idx)
        
        return min_diff