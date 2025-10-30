from typing import List
from collections import Counter

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        sum_total = sum(nums)
        count = Counter(nums)
        candidates = []
        
        for x in count:
            if (sum_total - x) % 2 != 0:
                continue
            y = (sum_total - x) // 2
            if y not in count:
                continue
            if x == y:
                if count[x] >= 2:
                    candidates.append(x)
            else:
                if count[y] >= 1:
                    candidates.append(x)
        
        return max(candidates)