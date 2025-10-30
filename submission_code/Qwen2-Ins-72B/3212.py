from collections import Counter
from typing import List

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10**9 + 7
        last_occurrence = {num: i for i, num in enumerate(nums)}
        max_in_prefix = [0] * n
        max_in_prefix[0] = nums[0]
        for i in range(1, n):
            max_in_prefix[i] = max(max_in_prefix[i-1], nums[i])
        
        partitions = 1
        for i in range(n):
            if max_in_prefix[i] <= i and i == n-1 or max_in_prefix[i] <= i and i < n-1 and last_occurrence[nums[i]] == i:
                partitions = (partitions * 2) % mod
        return partitions