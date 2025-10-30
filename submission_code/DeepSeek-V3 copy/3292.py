from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        # Precompute the last occurrence of each index in changeIndices
        last_occurrence = {}
        for s in range(m):
            idx = changeIndices[s]
            last_occurrence[idx] = s + 1  # 1-based seconds
        
        # Check if all indices have at least one occurrence in changeIndices
        for i in range(1, n+1):
            if i not in last_occurrence:
                return -1
        
        # Binary search for the earliest second
        left = 1
        right = m
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            # Check if it's possible to mark all indices by mid
            # We need to ensure that for each index i, we have enough time to decrement nums[i] to 0 and mark it
            # The marking must happen at or before its last occurrence in the first mid seconds
            possible = True
            total_time_needed = 0
            for i in range(1, n+1):
                # The last occurrence of i in the first mid seconds
                last = last_occurrence[i]
                if last > mid:
                    possible = False
                    break
                # Time needed to decrement nums[i-1] to 0
                decrement_time = nums[i-1]
                # The marking must happen at last, so the decrementing must be done before last
                if decrement_time > last - 1:
                    possible = False
                    break
                total_time_needed += decrement_time + 1  # +1 for marking
            if possible and total_time_needed <= mid:
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result