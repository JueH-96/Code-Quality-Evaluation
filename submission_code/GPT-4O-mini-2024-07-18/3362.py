from typing import List
from collections import defaultdict

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        uniqueness = []
        
        for i in range(n):
            distinct_count = 0
            freq = defaultdict(int)
            for j in range(i, n):
                if freq[nums[j]] == 0:
                    distinct_count += 1
                freq[nums[j]] += 1
                uniqueness.append(distinct_count)
        
        uniqueness.sort()
        length = len(uniqueness)
        
        if length % 2 == 1:
            return uniqueness[length // 2]
        else:
            return uniqueness[length // 2 - 1]