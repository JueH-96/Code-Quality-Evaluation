from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        # Find all known values and their positions
        known_values = [num for num in nums if num != -1]
        
        # If there are no known values, we can choose x = y and get differences of 0
        if not known_values:
            return 0
        
        # Find the maximum difference between consecutive known elements without -1s between them
        max_consecutive_diff = 0
        i = 0
        while i < len(nums):
            if nums[i] != -1:
                current = nums[i]
                j = i + 1
                while j < len(nums) and nums[j] == -1:
                    j += 1
                if j < len(nums) and nums[j] != -1:
                    max_consecutive_diff = max(max_consecutive_diff, abs(nums[j] - current))
                i = j
            else:
                i += 1
        
        # Binary search over D
        left = 0
        right = max(max_consecutive_diff, max(known_values) - min(known_values))
        min_D = right
        
        global_min = min(known_values)
        global_max = max(known_values)
        
        while left <= right:
            mid = (left + right) // 2
            # Check if all consecutive known differences without -1s are <= mid
            # We already have max_consecutive_diff, so if mid < max_consecutive_diff, it's not feasible
            if mid < max_consecutive_diff:
                left = mid + 1
                continue
            # Compute the intersection range [max_min, min_max]
            max_min = global_min - mid
            min_max = global_max + mid
            # Check if min_max - max_min >= mid
            if min_max - max_min >= mid:
                min_D = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return min_D