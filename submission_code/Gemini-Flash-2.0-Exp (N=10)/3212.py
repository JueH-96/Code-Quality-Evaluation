class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        last_occurrence = {}
        for i, num in enumerate(nums):
            last_occurrence[num] = i
        
        max_reach = 0
        count = 0
        
        for i in range(n):
            max_reach = max(max_reach, last_occurrence[nums[i]])
            if i == max_reach:
                count += 1
        
        if count == 0:
            return 0
        
        return pow(2, count - 1, MOD)