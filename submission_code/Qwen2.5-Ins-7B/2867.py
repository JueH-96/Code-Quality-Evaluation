class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        prev_one = -1
        count = 1
        
        for i, num in enumerate(nums):
            if num == 1:
                if prev_one != -1:
                    count *= (i - prev_one)
                    count %= MOD
                prev_one = i
        
        return count if prev_one != -1 else 0