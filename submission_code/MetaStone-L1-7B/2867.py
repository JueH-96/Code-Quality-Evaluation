class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        ones = [i for i, x in enumerate(nums) if x == 1]
        if not ones:
            return 0
        res = 1
        for i in range(len(ones) - 1):
            res *= (ones[i+1] - ones[i])
            res %= MOD
        return res