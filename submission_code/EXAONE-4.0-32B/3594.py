from collections import Counter
from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        total = sum(nums)
        freq = Counter(nums)
        candidates = set()
        
        for num in nums:
            x = total - 2 * num
            if num == x:
                if freq[num] >= 2:
                    candidates.add(x)
            else:
                if x in freq and freq[x] > 0:
                    candidates.add(x)
                    
        return max(candidates)