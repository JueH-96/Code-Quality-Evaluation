from collections import Counter
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        freq = Counter(nums)
        max_len = 0
        
        # Handle the special case for x = 1
        if 1 in freq:
            cnt = freq[1]
            possible = cnt if cnt % 2 == 1 else cnt - 1
            max_len = possible
        
        # Process other elements
        for x in freq:
            if x == 1:
                continue
            current_val = x
            m = 0
            current_m = 0
            while True:
                next_val = current_val * current_val
                if next_val > 10**9:
                    break
                if next_val in freq and freq[current_val] >= 2:
                    m += 1
                    current_val = next_val
                    current_m = m
                else:
                    break
            current_length = 2 * current_m + 1
            if current_length > max_len:
                max_len = current_length
        
        return max_len