from typing import List

class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        count = 0
        for num in nums1:
            for j in nums2:
                divisor = j * k
                if num % divisor == 0:
                    count += 1
        return count