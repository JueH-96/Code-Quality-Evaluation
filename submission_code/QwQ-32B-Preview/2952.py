from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        sum_nums1 = sum(nums1)
        sum_nums2 = sum(nums2)
        
        # For each t from 0 to n, compute the sum of top t gains
        for t in range(n + 1):
            # Calculate the required sum of gains
            required = sum_nums1 + t * sum_nums2 - x
            if required <= 0:
                return t
            # Calculate gains for current t
            gains = [nums1[i] + t * nums2[i] for i in range(n)]
            # Sort gains in descending order
            gains_sorted = sorted(gains, reverse=True)
            # Sum of top t gains
            sum_gains = sum(gains_sorted[:t])
            if sum_gains >= required:
                return t
        return -1