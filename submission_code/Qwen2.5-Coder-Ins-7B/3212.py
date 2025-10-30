class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        last = {}
        for i, num in enumerate(nums):
            last[num] = i
        
        partitions = 0
        end = 0
        for i, num in enumerate(nums):
            end = max(end, last[num])
            if i == end:
                partitions += 1
        
        return pow(2, partitions - 1, MOD)