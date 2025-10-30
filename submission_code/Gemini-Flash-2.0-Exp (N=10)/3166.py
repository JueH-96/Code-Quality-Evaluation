from collections import Counter
from math import ceil

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        counts = Counter(nums)
        freqs = list(counts.values())
        
        def check(k):
            groups = 0
            for freq in freqs:
                if freq < k:
                    return float('inf')
                
                groups += ceil(freq / (k + 1))
            return groups
        
        
        min_groups = float('inf')
        for k in range(1, max(freqs) + 1):
            groups = check(k)
            min_groups = min(min_groups, groups)
        
        return min_groups