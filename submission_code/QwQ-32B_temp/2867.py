from typing import List

class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        ones = [i for i, num in enumerate(nums) if num == 1]
        if len(ones) < 1:
            return 0
        if len(ones) == 1:
            return 1
        result = 1
        for i in range(1, len(ones)):
            result = result * (ones[i] - ones[i-1]) % MOD
        return result