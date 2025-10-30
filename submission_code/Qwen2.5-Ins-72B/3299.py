from collections import Counter
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        count = Counter(nums)
        nums_set = set(nums)
        nums_set.discard(1)
        
        def find_max_length(x):
            length = 0
            while x in nums_set:
                length += 2
                x *= x
            return length - 1
        
        max_length = 1
        for num in nums_set:
            max_length = max(max_length, find_max_length(num))
        
        if count[1] > 0:
            max_length = max(max_length, count[1] - (1 if count[1] % 2 == 0 else 0))
        
        return max_length