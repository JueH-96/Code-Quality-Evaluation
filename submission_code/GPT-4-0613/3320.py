from typing import List
from collections import Counter

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        counter = Counter(nums)
        operations = Counter()
        for num in sorted(counter.keys()):
            while counter[num] > 1:
                counter[num] -= 2
                operations[num * 2] += 1
            if counter[num] == 1 and counter[num * 2] > 0:
                counter[num] = 0
                counter[num * 2] -= 1
                operations[num * 2] += 1
        return max(operations.values()) if operations else 0