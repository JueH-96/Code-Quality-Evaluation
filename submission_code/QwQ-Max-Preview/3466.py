from collections import defaultdict
from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        current_counts = defaultdict(int)
        result = 0
        for num in nums:
            temp = defaultdict(int)
            for val, cnt in current_counts.items():
                new_val = val & num
                temp[new_val] += cnt
            temp[num] += 1
            current_counts = temp
            result += current_counts.get(k, 0)
        return result