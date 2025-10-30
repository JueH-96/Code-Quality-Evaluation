class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        indices = [i for i, num in enumerate(nums) if num == 1]
        if not indices:
            return 0
        result = 1
        for i in range(1, len(indices)):
            result = (result * (indices[i] - indices[i - 1])) % MOD
        return result