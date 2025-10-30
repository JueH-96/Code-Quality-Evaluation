from collections import defaultdict
from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        total = 0
        prev_and = defaultdict(int)
        for num in nums:
            curr_and = defaultdict(int)
            for and_val, count in prev_and.items():
                new_and = and_val & num
                curr_and[new_and] += count
            curr_and[num] += 1
            prev_and = curr_and
            total += curr_and.get(k, 0)
        return total