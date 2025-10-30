class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        last_occurrence = {}
        for i, num in enumerate(nums):
            last_occurrence[num] = i
        
        segments = 0
        max_end = 0
        
        for i, num in enumerate(nums):
            max_end = max(max_end, last_occurrence[num])
            if i == max_end:
                segments += 1
        
        mod = 10**9 + 7
        return pow(2, segments - 1, mod)