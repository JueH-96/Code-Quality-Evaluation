from typing import List
from collections import defaultdict

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        count = defaultdict(int)
        distinct_counts = []
        
        for i in range(n):
            for j in range(i, n):
                count[nums[j]] += 1
                distinct_counts.append(len(count))
        
        distinct_counts.sort()
        mid = len(distinct_counts) // 2
        return distinct_counts[mid]