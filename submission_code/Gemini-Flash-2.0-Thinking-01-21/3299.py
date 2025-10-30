from collections import Counter
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        counts = Counter(nums)
        max_len = counts.get(1, 0)
        unique_nums = sorted(list(set(nums)))
        for x in unique_nums:
            if x == 1:
                continue
            powers = [x]
            while powers[-1] <= 10**9:
                next_power = powers[-1] * powers[-1]
                if next_power > 10**9:
                    break
                powers.append(next_power)
            for l in range(len(powers), 0, -1):
                increasing_part = powers[:l]
                full_pattern = increasing_part + increasing_part[:-1][::-1] if l > 1 else increasing_part
                needed_counts = Counter(full_pattern)
                possible_to_form = True
                for val, count in needed_counts.items():
                    if counts.get(val, 0) < count:
                        possible_to_form = False
                        break
                if possible_to_form:
                    max_len = max(max_len, len(full_pattern))
                    break
        return max_len