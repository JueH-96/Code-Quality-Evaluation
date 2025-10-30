import math
from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Compute M_fixed: max absolute difference between adjacent fixed elements
        max_fixed = 0
        for i in range(n - 1):
            if nums[i] != -1 and nums[i + 1] != -1:
                max_fixed = max(max_fixed, abs(nums[i] - nums[i + 1]))
        
        # Find all gaps (blocks of consecutive -1s)
        gaps = []
        i = 0
        while i < n:
            if nums[i] == -1:
                start_idx = i
                # Find the end of the block of -1s
                while i < n and nums[i] == -1:
                    i += 1
                end_idx = i - 1
                # Determine left and right neighboring values
                L = None
                if start_idx > 0 and nums[start_idx - 1] != -1:
                    L = nums[start_idx - 1]
                R = None
                if end_idx < n - 1 and nums[end_idx + 1] != -1:
                    R = nums[end_idx + 1]
                gaps.append((L, R))
            else:
                i += 1
        
        # Define the check function for a given D
        def check(D: int) -> bool:
            if D < max_fixed:
                return False
            # Check for each gap with both L and R if |L - R| <= 2 * D
            for L_val, R_val in gaps:
                if L_val is not None and R_val is not None:
                    if abs(L_val - R_val) > 2 * D:
                        return False
            # Create list of intervals for gaps with at least one constraint
            intervals = []
            for L_val, R_val in gaps:
                if L_val is not None or R_val is not None:
                    if L_val is not None and R_val is not None:
                        a = max(L_val - D, R_val - D)
                        b = min(L_val + D, R_val + D)
                    elif L_val is not None:
                        a = L_val - D
                        b = L_val + D
                    elif R_val is not None:
                        a = R_val - D
                        b = R_val + D
                    intervals.append((a, b))
            # If no intervals, return True
            if not intervals:
                return True
            # Sort intervals by end time
            intervals.sort(key=lambda x: x[1])
            # Greedy algorithm to find the size of the maximum disjoint set
            count = 0
            current_end = float('-inf')
            for start, end in intervals:
                if start > current_end:
                    count += 1
                    current_end = end
            return count <= 2
        
        # Binary search for the minimum D
        low = 0
        high = 2000000000  # 2e9
        while low <= high:
            mid = (low + high) // 2
            if check(mid):
                high = mid - 1
            else:
                low = mid + 1
        return low