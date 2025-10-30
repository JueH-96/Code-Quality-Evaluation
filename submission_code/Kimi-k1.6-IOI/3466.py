from collections import defaultdict
from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        total = 0
        prev_counts = defaultdict(int)
        for num in nums:
            current_counts = defaultdict(int)
            for and_val, cnt in prev_counts.items():
                new_and = and_val & num
                current_counts[new_and] += cnt
            current_counts[num] += 1
            total += current_counts[k]
            prev_counts = current_counts
        return total