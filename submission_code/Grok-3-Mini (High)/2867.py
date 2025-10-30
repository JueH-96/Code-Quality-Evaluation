class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 1000000007
        result = 1
        prev_idx = -1
        one_count = 0
        for i, num in enumerate(nums):
            if num == 1:
                if prev_idx != -1:
                    diff = i - prev_idx
                    result = (result * diff) % MOD
                prev_idx = i
                one_count += 1
        if one_count == 0:
            return 0
        return result