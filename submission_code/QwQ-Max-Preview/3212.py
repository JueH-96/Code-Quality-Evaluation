class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        last_occurrence = {}
        for idx, num in enumerate(nums):
            last_occurrence[num] = idx
        
        m = 0
        current_end = 0
        start = 0
        
        for i in range(len(nums)):
            current_end = max(current_end, last_occurrence[nums[i]])
            if i == current_end:
                m += 1
                start = i + 1
        
        return pow(2, m - 1, MOD)