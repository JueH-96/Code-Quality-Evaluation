from typing import List

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        last_occurrence = {num: i for i, num in enumerate(nums)}
        partitions = 0
        max_end = 0
        for i in range(len(nums)):
            max_end = max(max_end, last_occurrence[nums[i]])
            if i == max_end:
                partitions += 1
        
        MOD = 10**9 + 7
        return pow(2, partitions - 1, MOD)