from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        from collections import Counter
        count = Counter(nums)
        operations = 0
        for val in count.values():
            if val == 1:
                return -1
            operations += val // 3 + (1 if val % 3 else 0)
        return operations