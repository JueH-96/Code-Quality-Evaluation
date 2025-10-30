from typing import List

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        last_occurrence = {num: i for i, num in enumerate(nums)}
        count = 0
        end = -1
        
        for i, num in enumerate(nums):
            if i > end:
                count += 1
            end = max(end, last_occurrence[num])
        
        return pow(2, count - 1, MOD)