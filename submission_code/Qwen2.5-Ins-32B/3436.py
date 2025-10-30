from typing import List
from itertools import accumulate

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        or_set = {0}
        min_diff = float('inf')
        
        for num in nums:
            or_set |= {num | x for x in or_set}
            min_diff = min(min_diff, min(abs(k - x) for x in or_set))
        
        return min_diff