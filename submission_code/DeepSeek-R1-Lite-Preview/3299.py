from typing import List
from collections import Counter

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        freq = Counter(nums)
        unique_nums = sorted(freq.keys())
        max_length = 1  # At least one element can always be selected

        for x in unique_nums:
            chain = []
            current = x
            while current in freq:
                chain.append(current)
                if current == 1:
                    break
                next_val = current * current
                if next_val > 10**18:  # Prevent overflow and unrealistic values
                    break
                current = next_val
            # Determine pattern size
            pattern_size = 2 * len(chain) - 1
            # Check frequency requirements
            valid = True
            for i in range(len(chain) - 1):
                if freq[chain[i]] < 2:
                    valid = False
                    break
            if valid and freq[chain[-1]] >= 1:
                if pattern_size > max_length:
                    max_length = pattern_size
        return max_length