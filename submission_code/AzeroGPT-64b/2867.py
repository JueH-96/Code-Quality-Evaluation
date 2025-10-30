class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        cnt, last = 0, -1 
        for i, x in enumerate(nums):
            if x: 
                if cnt: last += 1 
                else: last = i; cnt += 1 
        if cnt == 0: return 0 
        return pow(2, last - cnt + 1, 1_000_000_007)