from typing import List

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        last_occurrence = {}
        for i, num in enumerate(nums):
            last_occurrence[num] = i
        
        partitions = 1
        start, end = 0, 0
        for i, num in enumerate(nums):
            end = max(end, last_occurrence[num])
            if i == end:
                partitions *= 2
                partitions %= 1000000007
        return partitions