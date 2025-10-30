from typing import List
from collections import Counter

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def normalize(num):
            num_str = str(num)
            if len(num_str) == 1:
                return num_str
            count = Counter(num_str)
            sorted_digits = sorted(count.items())
            return ''.join(f"{digit}{count}" for digit, count in sorted_digits)
        
        normalized_nums = [normalize(num) for num in nums]
        count = Counter(normalized_nums)
        result = 0
        
        for c in count.values():
            if c > 1:
                result += c * (c - 1) // 2
        
        return result