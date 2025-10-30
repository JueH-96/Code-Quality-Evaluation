from typing import List
from collections import Counter

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        count = Counter(nums)
        max_len = 1 if 1 in count else 0
        
        for num in count:
            if num == 1:
                continue
            current_len = 0
            while num in count and count[num] >= 2:
                current_len += 2
                num *= num
            if num in count:
                current_len += 1
            else:
                current_len -= 1
            max_len = max(max_len, current_len)
        
        return max_len