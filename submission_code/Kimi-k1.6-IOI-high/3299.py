from collections import Counter
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        freq = Counter(nums)
        max_len = 1
        
        # Handle the special case for x = 1
        count_1 = freq.get(1, 0)
        if count_1 > 0:
            possible_m = (count_1 - 1) // 2
            max_len = 2 * possible_m + 1
        
        # Check all other elements
        for x in freq:
            if x == 1:
                continue
            current = x
            next_val = x * x
            possible_m = 0
            while True:
                # Check if next_val exists and current has enough frequency
                if next_val not in freq:
                    break
                if freq[current] < 2:
                    break
                possible_m += 1
                current = next_val
                next_val *= next_val  # next_val becomes current^2 for next iteration
            
            current_length = 2 * possible_m + 1
            if current_length > max_len:
                max_len = current_length
        
        return max_len