class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        prev = -1
        count = 0
        
        for i, num in enumerate(nums):
            if num == 1:
                if prev != -1:
                    count = (count * (i - prev)) % MOD
                prev = i
        
        return count if prev != -1 else 0