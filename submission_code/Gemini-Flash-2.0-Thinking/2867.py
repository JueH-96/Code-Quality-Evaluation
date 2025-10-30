class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        indices = [i for i, num in enumerate(nums) if num == 1]
        n_ones = len(indices)

        if n_ones < 2:
            return 0

        MOD = 10**9 + 7
        ways = 1

        for i in range(n_ones - 1):
            diff = indices[i + 1] - indices[i]
            ways = (ways * diff) % MOD

        return ways