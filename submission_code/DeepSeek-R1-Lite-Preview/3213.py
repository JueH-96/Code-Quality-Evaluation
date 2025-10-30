from typing import List
from collections import defaultdict

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        unique_m = sorted(set(nums), reverse=True)
        total = 0
        
        # Precompute for each m, the positions where nums[i] == m
        pos = defaultdict(list)
        for idx, val in enumerate(nums):
            pos[val].append(idx)
        
        # To keep track of segments where all elements <= m
        # We can use the positions where nums[i] > m as segment separators
        # But for efficiency, we'll directly work with the entire array for each m
        for m in unique_m:
            # Find all segments where all elements are <= m
            # Since m is decreasing, segments for higher m are supersets of lower m
            # For simplicity, consider the entire array as one segment for each m
            # As no element > m exists in the segment
            # Positions of m in the segment
            positions = pos[m]
            t = len(positions)
            if t < k:
                continue  # Not enough m's to form a subarray with at least k m's
            # Apply sliding window within the segment
            # Since the entire array is considered as the segment
            # We need to count subarrays with at least k m's
            left = 0
            count_m = 0
            for right in range(n):
                if nums[right] == m:
                    count_m += 1
                while count_m >= k:
                    total += n - right
                    if nums[left] == m:
                        count_m -= 1
                    left += 1
        return total