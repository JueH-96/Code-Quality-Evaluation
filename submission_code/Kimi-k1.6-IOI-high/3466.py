from collections import defaultdict
from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        total = 0
        prev_map = defaultdict(int)
        for num in nums:
            current_map = defaultdict(int)
            current_map[num] += 1
            for and_val, cnt in prev_map.items():
                new_and = and_val & num
                current_map[new_and] += cnt
            total += current_map[k]
            prev_map = current_map
        return total