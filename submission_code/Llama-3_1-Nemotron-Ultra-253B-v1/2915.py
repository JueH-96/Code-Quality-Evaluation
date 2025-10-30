from typing import List
from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        current_count = 0
        count_map = defaultdict(int)
        count_map[0] = 1  # Initial prefix sum is 0
        result = 0
        for num in nums:
            if num % modulo == k:
                current_count += 1
            current_mod = current_count % modulo
            target_mod = (current_mod - k) % modulo
            result += count_map.get(target_mod, 0)
            count_map[current_mod] += 1
        return result