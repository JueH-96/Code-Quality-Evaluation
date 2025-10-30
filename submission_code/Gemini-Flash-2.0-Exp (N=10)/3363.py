from typing import List
from collections import defaultdict

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        n = len(nums)
        ans = []
        counts = defaultdict(int)
        
        for i in range(n):
            counts[nums[i]] += freq[i]
            
            max_freq = 0
            if not counts:
                ans.append(0)
                continue
            
            for count in counts.values():
                max_freq = max(max_freq, count)
            
            ans.append(max_freq)
        
        return ans