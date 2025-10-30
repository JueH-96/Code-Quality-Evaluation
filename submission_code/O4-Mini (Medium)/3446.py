from typing import List

class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        count = 0
        # For each pair (i, j), check divisibility condition
        for a in nums1:
            for b in nums2:
                if a % (b * k) == 0:
                    count += 1
        return count