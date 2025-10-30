class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        first_one = -1
        last_one = -1
        count = 0
        
        for i, num in enumerate(nums):
            if num == 1:
                if first_one == -1:
                    first_one = i
                last_one = i
                if count > 0:
                    count = (count * (i - prev_one)) % MOD
                prev_one = i
        
        if first_one == -1:
            return 0
        
        return count if count > 0 else 1