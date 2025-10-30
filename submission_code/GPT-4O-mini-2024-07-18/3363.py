from collections import defaultdict
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        count = defaultdict(int)
        max_freq = 0
        ans = []
        
        for i in range(len(nums)):
            count[nums[i]] += freq[i]
            if count[nums[i]] > max_freq:
                max_freq = count[nums[i]]
            ans.append(max_freq if max_freq > 0 else 0)
        
        return ans