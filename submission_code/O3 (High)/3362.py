from typing import List
from collections import defaultdict

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        # helper: number of sub-arrays that contain at most k distinct values
        def subarrays_at_most(k: int) -> int:
            if k == 0:
                return 0
            freq = defaultdict(int)
            left = 0
            distinct = 0
            total  = 0
            for right, value in enumerate(nums):
                if freq[value] == 0:
                    distinct += 1
                freq[value] += 1

                while distinct > k:          # shrink until the window is valid
                    left_val = nums[left]
                    freq[left_val] -= 1
                    if freq[left_val] == 0:
                        distinct -= 1
                    left += 1
                # every sub-array that ends at `right` and starts anywhere in
                # [left .. right] is valid => (right - left + 1) of them
                total += right - left + 1
            return total

        n = len(nums)
        total_sub = n * (n + 1) // 2            # total number of sub-arrays
        target    = (total_sub + 1) // 2        # position of the median (1-based)
        max_dist  = len(set(nums))              # the largest possible uniqueness

        lo, hi = 1, max_dist
        answer = max_dist
        while lo <= hi:                         # binary search for smallest k with
            mid = (lo + hi) // 2                # at least `target` qualifying sub-arrays
            if subarrays_at_most(mid) >= target:
                answer = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return answer