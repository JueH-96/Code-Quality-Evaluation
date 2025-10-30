from collections import Counter
from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        total = sum(nums)
        count_map = Counter(nums)
        candidates = set()
        
        for O in nums:
            if (total - O) % 2 != 0:
                continue
            S = (total - O) // 2
            if S not in count_map:
                continue
            if O == S:
                if count_map[S] < 2:
                    continue
            candidates.add(O)
        
        return max(candidates)