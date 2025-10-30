from typing import List
from collections import Counter

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        suffix_count = Counter(nums)
        prefix_count = Counter()
        diff = []
        
        for i in range(n):
            prefix_count[nums[i]] += 1
            suffix_count[nums[i]] -= 1
            if suffix_count[nums[i]] == 0:
                del suffix_count[nums[i]]
            diff.append(len(prefix_count) - len(suffix_count))
        
        return diff