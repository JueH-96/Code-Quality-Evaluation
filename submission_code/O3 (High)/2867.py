from typing import List

MOD = 10**9 + 7

class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        # collect indices of all ones
        ones = [i for i, v in enumerate(nums) if v == 1]
        
        # if there is no 1 we cannot form any good subarray
        if not ones:
            return 0
        
        # one segment by default; multiply by choices between consecutive ones
        res = 1
        for i in range(1, len(ones)):
            gap = ones[i] - ones[i - 1] - 1      # number of zeros between consecutive ones
            res = (res * (gap + 1)) % MOD        # (gap + 1) ways to cut in this gap
        
        return res