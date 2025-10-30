from typing import List

class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        count_ones = nums.count(1)

        if count_ones == 0:
            return 0

        # Initialize variables
        result = 1
        prev_index = -1

        for i in range(n):
            if nums[i] == 1:
                if prev_index != -1:
                    result = result * (i - prev_index) % MOD
                prev_index = i

        return result