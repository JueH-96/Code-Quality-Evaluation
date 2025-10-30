from typing import List

class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        count = 0
        for x in nums1:
            for y in nums2:
                if x % (y * k) == 0:
                    count += 1
        return count