from collections import Counter
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        counts = Counter(nums)
        unique_nums = sorted(counts.keys())
        max_length = 0

        for start_val in unique_nums:
            if start_val == 1:
                max_length = max(max_length, counts[1])
                continue

            powers = [start_val]
            current_power = start_val
            while True:
                next_power = current_power * current_power
                if next_power not in counts:
                    break
                powers.append(next_power)
                current_power = next_power

            for i in range(len(powers)):
                peak = powers[i]
                increasing_part = powers[:i + 1]
                decreasing_part = increasing_part[:-1][::-1]
                pattern = increasing_part + decreasing_part

                temp_counts = counts.copy()
                current_length = 0
                possible = True
                for val in pattern:
                    if temp_counts.get(val, 0) > 0:
                        temp_counts[val] -= 1
                        current_length += 1
                    else:
                        possible = False
                        break

                if possible:
                    max_length = max(max_length, current_length)

        return max_length