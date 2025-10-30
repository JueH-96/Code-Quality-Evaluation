from collections import defaultdict
from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        count = 0
        prefix_counts = defaultdict(int)
        prefix_counts[0] = 1
        prefix_sum = 0

        for num in nums:
            if num % modulo == k:
                prefix_sum += 1
            current_mod = prefix_sum % modulo
            target = (current_mod - k) % modulo
            count += prefix_counts[target]
            prefix_counts[current_mod] += 1

        return count