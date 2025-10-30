from typing import List

class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        count = 0
        prev = -1
        result = 1
        
        for i, num in enumerate(nums):
            if num == 1:
                if prev != -1:
                    result = (result * (i - prev)) % MOD
                prev = i
                count += 1
        
        if count == 0:
            return 0
        elif count == 1:
            return 1
        else:
            return result