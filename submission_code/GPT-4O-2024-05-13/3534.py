from typing import List
from collections import Counter

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def digit_count_key(num):
            return tuple(sorted(str(num)))
        
        count = Counter(digit_count_key(num) for num in nums)
        
        result = 0
        for key in count:
            n = count[key]
            result += n * (n - 1) // 2
        
        return result