from collections import Counter
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        counts = Counter(nums)
        unique_nums = sorted(list(set(nums)))
        max_len = 0
        for x in unique_nums:
            current_len = 0
            p = 0
            possible = True
            current_x = x
            while possible:
                length = 2 * p + 1
                powers_needed = {}
                valid_pattern = True
                for i in range(p):
                    power_val = current_x**(2**i)
                    if counts[power_val] < 2:
                        valid_pattern = False
                        break
                if not valid_pattern:
                    break
                peak_power = current_x**(2**p)
                if counts[peak_power] < 1:
                    valid_pattern = False
                    break
                if valid_pattern:
                    current_len = length
                    max_len = max(max_len, current_len)
                    p += 1
                else:
                    possible = False
                    
        if max_len == 0 and nums:
            return 1
        return max_len