class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        last_occurrence = {}
        for i, num in enumerate(nums):
            last_occurrence[num] = i
        
        count = 0
        end = 0
        for i in range(len(nums)):
            end = max(end, last_occurrence[nums[i]])
            if i == end:
                count += 1
        
        return pow(2, count - 1, MOD)