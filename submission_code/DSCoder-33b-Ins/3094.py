from typing import List
from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        counter = Counter(nums)
        two_count = 0
        three_count = 0
        for count in counter.values():
            while count >= 3 and count % 2 == 0:
                count -= 2
                two_count += 1
            while count >= 3:
                count -= 3
                three_count += 1
            if count != 0:
                return -1
        return two_count + three_count