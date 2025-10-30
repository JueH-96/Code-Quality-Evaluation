from collections import Counter
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        count = Counter(nums)
        total_operations = 0
        for c in count.values():
            if c == 1:
                return -1
            max_b = c // 3
            for b in range(max_b, -1, -1):
                rem = c - 3 * b
                if rem >= 0 and rem % 2 == 0:
                    total_operations += b + rem // 2
                    break
        return total_operations