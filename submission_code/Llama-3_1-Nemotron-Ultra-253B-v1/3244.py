from collections import Counter
from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        freq = Counter(nums)
        f = max(freq.values())
        sum_rest = len(nums) - f
        if f > sum_rest + 1:
            return (f - sum_rest - 1) // 2 + 1
        else:
            return 1