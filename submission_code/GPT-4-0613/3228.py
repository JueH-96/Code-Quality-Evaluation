from typing import List
from collections import Counter

class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        counter1 = Counter(nums1)
        counter2 = Counter(nums2)
        counter = counter1 + counter2
        counter = sorted(counter.items(), key=lambda x: -x[1])
        n = len(nums1)
        remove = n // 2
        result = 0
        for num, count in counter:
            if remove >= count:
                remove -= count
            else:
                result += count - remove
                remove = 0
            if remove == 0:
                break
        return result