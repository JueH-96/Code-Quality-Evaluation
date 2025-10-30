from collections import Counter
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        freq = Counter(nums)
        max_len = 0
        
        for x in freq:
            if x == 1:
                # Handle the case for 1
                cnt = freq[x]
                current_max = cnt if cnt % 2 == 1 else cnt - 1
                if current_max > max_len:
                    max_len = current_max
            else:
                current = x
                steps = 0
                while True:
                    next_num = current * current
                    # Check if current has enough count and next_num exists
                    if freq[current] >= 2 and next_num in freq:
                        steps += 1
                        current = next_num
                    else:
                        break
                # Calculate candidate
                candidate = 2 * steps + 1
                if candidate > max_len:
                    max_len = candidate
        
        return max_len