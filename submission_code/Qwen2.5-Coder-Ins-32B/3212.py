class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        last_occurrence = {num: idx for idx, num in enumerate(nums)}
        end = 0
        partitions = 0
        
        for i, num in enumerate(nums):
            end = max(end, last_occurrence[num])
            if i == end:
                partitions += 1
        
        return pow(2, partitions - 1, MOD)