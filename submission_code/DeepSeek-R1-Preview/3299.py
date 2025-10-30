from collections import Counter
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        freq = Counter(nums)
        max_len = 1  # At least one element can be chosen

        # Process all elements except 1
        for x in freq:
            if x == 1:
                continue
            current = x
            m = 0
            while True:
                next_x = current ** 2
                if next_x == current:
                    break
                if next_x in freq and freq[current] >= 2:
                    m += 1
                    current = next_x
                else:
                    break
            chain_len = 2 * m + 1
            if chain_len > max_len:
                max_len = chain_len

        # Handle the case for x = 1
        if 1 in freq:
            count = freq[1]
            chain_len = count if count % 2 == 1 else count - 1
            if chain_len > max_len:
                max_len = chain_len

        return max_len